kernel_code = """
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

__kernel void check_footprint_intersection(
    __global const float *trajectory,      // 2D trajectory points [x1, y1, theta1, x2, y2, theta2, ...]
    const float footprint_length,          // Footprint length
    const float footprint_width,           // Footprint width
    __global const float *obstacle,        // Obstacle vertices [x1, y1, x2, y2, ...]
    const int num_obstacle_vertices,       // Number of obstacle vertices
    __global int *results                  // Output: 1 if intersects, 0 otherwise
) {
    int id = get_global_id(0); // Each thread handles one trajectory point

    // Extract the pose (x, y, theta) of the current trajectory point
    float x = trajectory[id * 3 + 0];
    float y = trajectory[id * 3 + 1];
    float theta = trajectory[id * 3 + 2]; // Orientation in radians

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

    // Check intersection with the obstacle using the Separating Axis Theorem (SAT)
    int intersects = 1;

    // Combine the rectangle and obstacle vertices for axis generation
    float2 obstacle_vertices[100]; // Assuming a max of 100 vertices for the obstacle
    for (int i = 0; i < num_obstacle_vertices; i++) {
        obstacle_vertices[i] = (float2)(obstacle[i * 2], obstacle[i * 2 + 1]);
    }

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
    results[id] = intersects;
}
"""

import pyopencl as cl
import numpy as np


def next_power_of_2(n):
    if n == 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


# Initialize OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Define kernel
program = cl.Program(context, kernel_code).build()

# Example constants
footprint_length = 1.0
footprint_width = 0.5

# Example data 1
trajectory = np.array([[0, 0, 0], [1, 1, np.pi/4]], dtype=np.float32).flatten()
obstacle = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32).flatten()
num_obstacle_vertices = len(obstacle) // 2
results = np.zeros(len(trajectory) // 3, dtype=np.int32)

# Create buffers
trajectory_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=trajectory)
obstacle_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=obstacle)
results_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results.nbytes)

# Set arguments and execute kernel
globalSize = next_power_of_2(len(trajectory))
# localSize = 16
program.check_footprint_intersection(
    queue, (globalSize,), None, #(localSize,),
    trajectory_buffer, np.float32(footprint_length), np.float32(footprint_width),
    obstacle_buffer, np.int32(num_obstacle_vertices), results_buffer
)
cl.enqueue_copy(queue, results, results_buffer)

print("Results:", results)