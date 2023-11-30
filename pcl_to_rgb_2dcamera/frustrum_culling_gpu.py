import numpy as np
import pyopencl as cl
import time

from chunksSplit import split_array_indices

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

# Define OpenCL kernel for frustum culling
kernel_code = """
__kernel void frustum_culling(__global float* points, __global int* result, __global float* projection_matrix, __global float* camera_position) {
    int gid = get_global_id(0);

    float4 homogeneous_point;
    homogeneous_point.x = points[gid*3] - camera_position[0];
    homogeneous_point.y = points[gid*3+1] - camera_position[1];
    homogeneous_point.z = points[gid*3+2] - camera_position[2];
    homogeneous_point.w = 1.0f;
    
    // printf("%u\\n", gid);

    // Print intermediate results for debugging
    // printf(" [%u] homogeneous_point: (%f, %f, %f, %f)\\n", gid, homogeneous_point.x, homogeneous_point.y, homogeneous_point.z, homogeneous_point.w);

    float4 eye_space_point;
    eye_space_point.x = projection_matrix[0] * homogeneous_point.x + projection_matrix[1] * homogeneous_point.y + projection_matrix[2] * homogeneous_point.z + projection_matrix[3] * homogeneous_point.w;
    eye_space_point.y = projection_matrix[4] * homogeneous_point.x + projection_matrix[5] * homogeneous_point.y + projection_matrix[6] * homogeneous_point.z + projection_matrix[7] * homogeneous_point.w;
    eye_space_point.z = projection_matrix[8] * homogeneous_point.x + projection_matrix[9] * homogeneous_point.y + projection_matrix[10] * homogeneous_point.z + projection_matrix[11] * homogeneous_point.w;
    eye_space_point.w = projection_matrix[12] * homogeneous_point.x + projection_matrix[13] * homogeneous_point.y + projection_matrix[14] * homogeneous_point.z + projection_matrix[15] * homogeneous_point.w;
    
    // Print intermediate results for debugging
    // printf(" [%u] eye_space_point: (%f, %f, %f, %f)\\n", gid, eye_space_point.x, eye_space_point.y, eye_space_point.z, eye_space_point.w);

    float xw = eye_space_point.x/eye_space_point.w;
    float yw = eye_space_point.y/eye_space_point.w;
    float zw = eye_space_point.z/eye_space_point.w;
    // printf(" [%u] xy,yw,zw: (%f, %f, %f)\\n", gid, xw, yw, zw);

    // Check if the point is inside the view frustum
    if (-1.0f <= xw && xw <= 1.0f &&
        -1.0f <= yw && yw <= 1.0f &&
        -1.0f <= zw && zw <= 1.0f) {
        result[gid] = 1;
    }else{
        result[gid] = 0;
    }
}
"""

def filter_points_by_frustum_opencl(points_3d, camera_position, fov, aspect_ratio, near, far):
    # Set up OpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    ### calculate amount of queries needed for all points
    a,b,c = device.max_work_item_sizes
    N = a*b 
    chunks = int(np.floor(len(points_3d) / N) + 1)
    chunk_ranges = split_array_indices(len(points_3d), chunks)
    filtered_points = np.empty_like(points_3d)
    for start_index, end_index in chunk_ranges:
        points = points_3d[start_index:end_index]
        program = cl.Program(context, kernel_code).build()
        # Set up OpenCL buffers
        points_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(points).flatten().astype(np.float32))
        result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, len(points) * np.int8().itemsize)
        ### set up camera projection matrix
        f = 1 / np.tan(0.5 * np.radians(fov))
        projection_matrix = np.zeros((4,4), dtype=np.float32)
        projection_matrix[0,0] = f / aspect_ratio
        projection_matrix[1,1] = f
        projection_matrix[2,2] = (far + near) / (near - far)
        projection_matrix[2,3] = 2.0 * far * near / (near - far)
        projection_matrix[3,2] = -1.0
        camera_position_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(camera_position).astype(np.float32))
        projection_matrix_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=projection_matrix.flatten().astype(np.float32))
        ### print debug buffer infos
        b1 = points_buffer.get_info(cl.mem_info.SIZE)
        b2 = result_buffer.get_info(cl.mem_info.SIZE)
        b3 = camera_position_buffer.get_info(cl.mem_info.SIZE)
        b4 = projection_matrix_buffer.get_info(cl.mem_info.SIZE)
        # Execute the kernel
        # program.frustum_culling(queue, (len(points_3d),), None, points_buffer, result_buffer, projection_matrix_buffer, camera_position_buffer)
        # global_size and local_size are tuples of identical length, with between one and three entries. 
        # global_size specifies the overall size of the computational grid: one work item will be launched for every integer point in the grid.
        # local_size specifies the workgroup size, which must evenly divide the global_size in a dimension-by-dimension manner. 
        # None may be passed for local_size, in which case the implementation will use an implementation-defined workgroup size. 
        # If g_times_l is True, the global size will be multiplied by the local size. (which makes the behavior more like Nvidia CUDA)
        # In this case, global_size and local_size also do not have to have the same number of entries.
        globalSize = next_power_of_2(b1+b2+b3+b4)
        localSize = 128
        # print(" MEM ===> %s" % globalSize)
        program.frustum_culling(queue, (globalSize,), (localSize,), points_buffer, result_buffer, projection_matrix_buffer, camera_position_buffer)

        # Read back the results
        result = np.empty(len(points), dtype=np.int8)
        cl.enqueue_copy(queue, result, result_buffer).wait()

        # Filter points based on the result and add to filtered_points
        correctPoints = points[result == 1]
        filtered_points = np.vstack([filtered_points, correctPoints])

    return filtered_points


# Example usage
camera_position = np.array([0.0, 0.0, 0.0])
fov = 60.0  # Field of view in degrees
aspect_ratio = 16/9  # Width/height ratio of the viewport
near = 0.1
far = 100.0

# Replace these with your actual 3D points
points_3d = np.random.rand(1000000, 3) * 20.0 - 10.0

for i in range(5):
    start = time.time()
    filtered_points_opencl = filter_points_by_frustum_opencl(points_3d, camera_position, fov, aspect_ratio, near, far)
    print(len(filtered_points_opencl))
    print("dt%s: %s" % (i, (time.time() - start)))

