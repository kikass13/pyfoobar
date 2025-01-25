#define MAX_OBSTACLE_VERTICES 10
#define COMBINED_DATA_LENGTH 5

// function to project polygon vertices onto an axis
float2 project(float2 *vertices, int num_vertices, float2 axis) {
    float min_proj = dot(vertices[0], axis);
    float max_proj = min_proj;
    for (int i = 1; i < num_vertices; i++) {
        float proj = dot(vertices[i], axis);
        if (proj < min_proj) min_proj = proj;
        if (proj > max_proj) max_proj = proj;
    }
    return (float2)(min_proj, max_proj);
}

// function to check if two projections overlap
int overlap(float2 proj1, float2 proj2) {
    return proj1.x <= proj2.y && proj2.x <= proj1.y;
}

// Helper function to check if the rotated rectangle intersects with an obstacle
// int check_intersection(float2 *rect, float2 *obstacle_vertices, int num_obstacle_vertices) {
//     // Apply the Separating Axis Theorem (SAT) as in the original kernel
//     // You can reuse the same projection and overlap logic for SAT here
//     // (Use the 'project' and 'overlap' functions as earlier)

//     // Return 1 if there's an intersection, 0 otherwise
//     // Check intersection with the obstacle using the Separating Axis Theorem (SAT)
//     int intersects = 1;

//     // Check all edges of both polygons as potential separating axes
//     int total_axes = num_obstacle_vertices + 4;
//     for (int i = 0; i < total_axes; i++) {
//         float2 edge = (i < 4) ? 
//             (rect[(i + 1) % 4] - rect[i]) : 
//             (obstacle_vertices[(i + 1 - 4) % num_obstacle_vertices] -
//              obstacle_vertices[i - 4]);

//         // Perpendicular axis
//         float2 axis = (float2)(-edge.y, edge.x);

//         // Project both polygons onto the axis
//         float2 rect_proj = project(rect, 4, axis);
//         float2 obstacle_proj = project(obstacle_vertices, num_obstacle_vertices, axis);

//         // If there's no overlap, no intersection exists
//         if (!overlap(rect_proj, obstacle_proj)) {
//             intersects = 0;
//             break;
//         }
//     }

//     // If no separating axis was found, the polygons intersect
//     return intersects;
// }

// Helper function to rotate a point by theta
float2 rotate_point(float2 point, float theta) {
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    return (float2)(point.x * cos_theta - point.y * sin_theta, point.x * sin_theta + point.y * cos_theta);
}

// Helper function to calculate the 4 corners of the robot's footprint
void calculate_footprint_corners(float x, float y, float theta, float length, float width, float2* corners) {
    // Half of the footprint length and width
    float half_length = length / 2.0f;
    float half_width = width / 2.0f;

    // Define the 4 corners of the footprint (in local coordinates)
    float2 local_corners[4] = {
        (float2)(-half_length, -half_width), // Bottom-left corner
        (float2)(half_length, -half_width),  // Bottom-right corner
        (float2)(half_length, half_width),   // Top-right corner
        (float2)(-half_length, half_width)   // Top-left corner
    };

    // Rotate the corners and translate to the robot's pose (x, y)
    for (int i = 0; i < 4; ++i) {
        corners[i] = rotate_point(local_corners[i], theta);  // Rotate each corner
        corners[i].x += x;  // Translate to robot's position (x, y)
        corners[i].y += y;
    }
}

// Helper function to check if a point is inside a polygon (point-in-polygon test)
bool point_in_polygon(float2 point, float2* polygon, int num_vertices) {
    bool inside = false;
    int j = num_vertices - 1;
    for (int i = 0; i < num_vertices; i++) {
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y) &&
            (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

// Helper function to check if the footprint (rectangle) intersects with the obstacle
bool check_collision(float2* footprint_corners, float2* obstacle_vertices, int num_vertices) {
    // Check if any of the footprint corners are inside the obstacle polygon
    for (int i = 0; i < 4; ++i) {
        if (point_in_polygon(footprint_corners[i], obstacle_vertices, num_vertices)) {
            return true;  // A corner is inside the obstacle polygon
        }
    }
    // If none of the corners are inside, we could implement additional checks (e.g., edge intersection)
    return false;
}

__kernel void check_footprint_collision(
    const float footprint_length,          // Length of the robot's footprint (rectangle)
    const float footprint_width,            // Width of the robot's footprint (rectangle)
    __global const float* combined_data,   // Flattened combined data: trajectory_pose and obstacle vertices
    const int num_pairs,                   // Total number of trajectory-obstacle pairs
    __global float *footprint_coords,      // Output: [num_trajectory_poses][4][2] (coordinates of footprint vertices)
    __global int* results                  // Array to store the collision results (1 for collision, 0 for no collision)
) {
    int pair_id = get_global_id(0);  // Get the index of the current pair (trajectory pose, obstacle)
    if (pair_id >= num_pairs) return;  // If the pair_id is out of range, return

    // Extract trajectory pose (x, y, theta)
    float x = combined_data[pair_id * COMBINED_DATA_LENGTH];  // x
    float y = combined_data[pair_id * COMBINED_DATA_LENGTH + 1];  // y
    float theta = combined_data[pair_id * COMBINED_DATA_LENGTH + 2];  // theta
    
    // Extract the number of vertices for the obstacle
    int num_vertices = (int)combined_data[pair_id * COMBINED_DATA_LENGTH + 3];  // num_vertices
    
    // Create an array to store the obstacle vertices (2D)
    float2 vertices[MAX_OBSTACLE_VERTICES];
    
    // Extract the obstacle vertices from the combined_data
    for (int i = 0; i < num_vertices; ++i) {
        int idx = pair_id * (COMBINED_DATA_LENGTH) + 4 + i * 2;  // Get the starting index of the vertices
        vertices[i] = (float2)(combined_data[idx], combined_data[idx + 1]);  // Store the vertex coordinates
    }

    // Calculate the 4 corners of the footprint (rectangle)
    float2 footprint_corners[4];
    calculate_footprint_corners(x, y, theta, footprint_length, footprint_width, footprint_corners);
    
    // Check for intersection between the footprint and the obstacle
    bool collision = check_collision(footprint_corners, vertices, num_vertices);

    footprint_coords[pair_id * 8 + 0] = footprint_corners[0].x;
    footprint_coords[pair_id * 8 + 1] = footprint_corners[0].y;
    footprint_coords[pair_id * 8 + 2] = footprint_corners[1].x;
    footprint_coords[pair_id * 8 + 3] = footprint_corners[1].y;
    footprint_coords[pair_id * 8 + 4] = footprint_corners[2].x;
    footprint_coords[pair_id * 8 + 5] = footprint_corners[2].y;
    footprint_coords[pair_id * 8 + 6] = footprint_corners[3].x;
    footprint_coords[pair_id * 8 + 7] = footprint_corners[3].y;

    // Store the result (1 if intersection found, 0 otherwise)
    results[pair_id] = collision;

}

