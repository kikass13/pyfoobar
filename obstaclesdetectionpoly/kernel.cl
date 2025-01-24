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
int check_intersection(float2 *rect, float2 *obstacle_vertices, int num_obstacle_vertices) {
    // Apply the Separating Axis Theorem (SAT) as in the original kernel
    // You can reuse the same projection and overlap logic for SAT here
    // (Use the 'project' and 'overlap' functions as earlier)

    // Return 1 if there's an intersection, 0 otherwise
    // Check intersection with the obstacle using the Separating Axis Theorem (SAT)
    int intersects = 1;

    // Check all edges of both polygons as potential separating axes
    int total_axes = num_obstacle_vertices + 4;
    for (int i = 0; i < total_axes; i++) {
        float2 edge = (i < 4) ? 
            (rect[(i + 1) % 4] - rect[i]) : 
            (obstacle_vertices[(i + 1 - 4) % num_obstacle_vertices] -
             obstacle_vertices[i - 4]);

        // Perpendicular axis
        float2 axis = (float2)(-edge.y, edge.x);

        // Project both polygons onto the axis
        float2 rect_proj = project(rect, 4, axis);
        float2 obstacle_proj = project(obstacle_vertices, num_obstacle_vertices, axis);

        // If there's no overlap, no intersection exists
        if (!overlap(rect_proj, obstacle_proj)) {
            intersects = 0;
            break;
        }
    }

    // If no separating axis was found, the polygons intersect
    return intersects;
}

__kernel void check_footprint_intersection(
    __global const float *trajectory,      // 2D trajectory points [x1, y1, theta1, x2, y2, theta2, ...]
    const int num_trajectory_poses,        // Number of trajectory poses
    __global const float *obstacles,       // All obstacle vertices in a flattened array [x1, y1, x2, y2, ...]
    __global const int *obstacle_sizes,    // Number of vertices for each obstacle
    const int num_obstacles,               // Total number of obstacles
    const float footprint_length,          // Footprint length
    const float footprint_width,           // Footprint width
    __global float *footprint_coords,      // Output: [num_trajectory_poses][4][2] (coordinates of footprint vertices)
    __global int *results                  // Output: [num_trajectory_poses][num_obstacles] (1 if intersects, 0 otherwise)
) {
    int obstacle_id = get_global_id(0); // Each thread handles one obstacle

    if (obstacle_id >= num_obstacles) return;

    // Get the number of vertices for the current obstacle
    int num_vertices = obstacle_sizes[obstacle_id];

    // Get the obstacle's vertices from the flattened array
    float2 obstacle_vertices[100]; // Assuming a max of 100 vertices per obstacle
    for (int i = 0; i < num_vertices; i++) {
        obstacle_vertices[i] = (float2)(obstacles[(obstacle_id * 100 + i) * 2], obstacles[(obstacle_id * 100 + i) * 2 + 1]);
    }

    // Loop over all trajectory poses
    for (int trajectory_id = 0; trajectory_id < num_trajectory_poses; trajectory_id++) {
        // Extract the pose (x, y, theta) of the current trajectory point
        float x = trajectory[trajectory_id * 3 + 0];
        float y = trajectory[trajectory_id * 3 + 1];
        float theta = trajectory[trajectory_id * 3 + 2]; // Orientation in radians

        // Calculate the vertices of the rotated rectangle (footprint)
        float half_length = footprint_length / 2.0f;
        float half_width = footprint_width / 2.0f;

        float cos_theta = cos(theta);
        float sin_theta = sin(theta);

        float2 rect[4];
        rect[0] = (float2)(x + cos_theta * half_length - sin_theta * half_width,
                           y + sin_theta * half_length + cos_theta * half_width);
        rect[1] = (float2)(x + cos_theta * half_length + sin_theta * half_width,
                           y + sin_theta * half_length - cos_theta * half_width);
        rect[2] = (float2)(x - cos_theta * half_length + sin_theta * half_width,
                           y - sin_theta * half_length - cos_theta * half_width);
        rect[3] = (float2)(x - cos_theta * half_length - sin_theta * half_width,
                           y - sin_theta * half_length + cos_theta * half_width);

        // Check if the rotated rectangle intersects the obstacle using Separating Axis Theorem (SAT)
        int intersects = check_intersection(rect, obstacle_vertices, num_vertices);

        int idx = trajectory_id * num_obstacles + obstacle_id;  // Index to store footprint coordinates
        footprint_coords[idx * 8 + 0] = rect[0].x;
        footprint_coords[idx * 8 + 1] = rect[0].y;
        footprint_coords[idx * 8 + 2] = rect[1].x;
        footprint_coords[idx * 8 + 3] = rect[1].y;
        footprint_coords[idx * 8 + 4] = rect[2].x;
        footprint_coords[idx * 8 + 5] = rect[2].y;
        footprint_coords[idx * 8 + 6] = rect[3].x;
        footprint_coords[idx * 8 + 7] = rect[3].y;
        
        // Store the result (1 if intersection found, 0 if not)
        results[idx] = intersects;
    }
}

