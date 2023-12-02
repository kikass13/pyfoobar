import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

kernel_code = """
__kernel void frustum_culling(__constant float* points, __constant float* frustum_planes, __global unsigned char* result) {
    int gid = get_global_id(0);

    int local_id = get_local_id(0);
    __local float3 localFrustumNormals[6];

    /// compute the frustrum normals once to speed up performance
    if (local_id == 0) {
        // Store the result in local memory
        for(unsigned int i = 0; i < 6; i++)
        {
            float3 frustrum_point0 = vload3(i*4 + 0, frustum_planes); // frustum_planes[i][0].x,y,z
            float3 frustrum_point1 = vload3(i*4 + 1, frustum_planes); // frustum_planes[i][1].x,y,z
            float3 frustrum_point2 = vload3(i*4 + 2, frustum_planes); // frustum_planes[i][2].x,y,z

            // compute normal of frustrum
            float3 l1 = frustrum_point1 - frustrum_point0;
            float3 l2 = frustrum_point2 - frustrum_point0;

            // cross product l1 x l2
            float3 normal = cross(l1, l2);
            
            // normalize normal vector (normal / length(normal) )
            normal = normalize(normal);
            localFrustumNormals[i] = normal;
        }
    }
    // Synchronize to make sure the result is stored before reading
    barrier(CLK_LOCAL_MEM_FENCE);

    // check point against every frustum normal
    float3 point = vload3(gid, points);
    for(unsigned int i = 0; i < 6; i++){
        // get a random point of the frustum plane 
        float3 frustrum_point0 = vload3(i*4 + 0, frustum_planes);

        // calculate point to reference on normal plane
        float3 vector_to_point = point - frustrum_point0;

        // dot product: vector_to_point . localFrustumNormals[i]
        float dotproduct = dot(vector_to_point, localFrustumNormals[i]);

        // Check if the point is inside the view frustum
        if (dotproduct < 0) {
            result[gid] = 0;
            break;
        }else{
            result[gid] = 1;
        }
    }
}
"""

class FrustumFilter:
    def __init__(self):
        # Create OpenCL context and queue
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.program = None
    def init(self):
        # Compile the OpenCL program
        self.program = cl.Program(self.context, kernel_code)
        self.program.build()
    def compute_frustum(self, observer_position, observer_direction, fov_degrees, near_clip, far_clip, aspect_ratio=1.0):
        # Convert field of view from degrees to radians
        fov_radians = np.radians(fov_degrees)
        # Compute frustum basis vectors
        forward = observer_direction / np.linalg.norm(observer_direction)
        right = np.cross([0, 0, -1], forward)
        up = -np.cross(forward, right)
        # Compute frustum near and far plane centers
        near_center = observer_position + forward * near_clip
        far_center = observer_position + forward * far_clip
        # Compute frustum half extents based on field of view
        near_height = np.tan(fov_radians / 2) * near_clip
        near_width = near_height * aspect_ratio
        far_height = np.tan(fov_radians / 2) * far_clip
        far_width = far_height * aspect_ratio
        # Compute frustum right, left, top, and bottom planes
        near_frustum_right = right * near_width
        near_frustum_left = -near_frustum_right
        near_frustum_top = up * near_height
        near_frustum_bottom = -near_frustum_top
        far_frustum_right = right * far_width
        far_frustum_left = -far_frustum_right
        far_frustum_top = up * far_height
        far_frustum_bottom = -far_frustum_top
        # Compute frustum points for near and far planes
        near_top_left = near_center + near_frustum_top + near_frustum_left
        near_top_right = near_center + near_frustum_top + near_frustum_right
        near_bottom_left = near_center + near_frustum_bottom + near_frustum_left
        near_bottom_right = near_center + near_frustum_bottom + near_frustum_right
        far_top_left = far_center + far_frustum_top + far_frustum_left
        far_top_right = far_center + far_frustum_top + far_frustum_right
        far_bottom_left = far_center + far_frustum_bottom + far_frustum_left
        far_bottom_right = far_center + far_frustum_bottom + far_frustum_right
        # Represent frustum planes as tuples of three points
        near_plane = (near_top_left, near_top_right, near_bottom_left, near_bottom_right)
        far_plane = (far_top_right, far_top_left, far_bottom_right, far_bottom_left)
        top_plane = (near_top_left, far_top_left, far_top_right, near_top_right)
        right_plane = (near_top_right, far_top_right, far_bottom_right, near_bottom_right)
        bottom_plane = (near_bottom_right, far_bottom_right, far_bottom_left, near_bottom_left)
        left_plane = (near_bottom_left, far_bottom_left, far_top_left, near_top_left)
        ### frustrum normals
        return np.array([near_plane, far_plane, top_plane, right_plane, bottom_plane, left_plane], dtype=np.float32)
    def filter_points_by_frustum_opencl(self, points_3d, frustum_planes):
        if self.program.get_build_info(self.device, cl.program_build_info.STATUS) > 0:
            raise ValueError("Kernel not compiled, please run .init() method")
        ### calculate amount of queries needed for all points
        a,b,c = self.device.max_work_item_sizes
        N = a*b 
        chunks = int(np.floor(len(points_3d) / N) + 1)
        chunk_ranges = split_array_indices(len(points_3d), chunks)
        filteredIndices = np.empty((len(points_3d)), dtype=np.int8)
        for start_index, end_index in chunk_ranges:
            points = points_3d[start_index:end_index]
            # Set up OpenCL buffers
            points_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(points).flatten().astype(np.float32))
            result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, len(points) * np.int8().itemsize)
            frustum_planes_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(frustum_planes).astype(np.float32))
            ### print debug buffer infos
            b1 = points_buffer.get_info(cl.mem_info.SIZE)
            b2 = result_buffer.get_info(cl.mem_info.SIZE)
            b3 = frustum_planes_buffer.get_info(cl.mem_info.SIZE)
            # Execute the kernel
            # program.frustum_culling(queue, (len(points_3d),), None, points_buffer, result_buffer, projection_matrix_buffer, camera_position_buffer)
            # global_size and local_size are tuples of identical length, with between one and three entries. 
            # global_size specifies the overall size of the computational grid: one work item will be launched for every integer point in the grid.
            # local_size specifies the workgroup size, which must evenly divide the global_size in a dimension-by-dimension manner. 
            # None may be passed for local_size, in which case the implementation will use an implementation-defined workgroup size. 
            # If g_times_l is True, the global size will be multiplied by the local size. (which makes the behavior more like Nvidia CUDA)
            # In this case, global_size and local_size also do not have to have the same number of entries.
            globalSize = next_power_of_2(b1+b2+b3)
            localSize = 128
            # print(" MEM ===> %s" % globalSize)
            self.program.frustum_culling(self.queue, (globalSize,), (localSize,), points_buffer, frustum_planes_buffer, result_buffer)
            # Read back the results
            result = np.empty(len(points), dtype=np.int8)
            cl.enqueue_copy(self.queue, result, result_buffer).wait()
            # Filter points based on the result and add to filtered_points
            filteredIndices[start_index:end_index] = result
        return filteredIndices
    def plot_frustum(self, ax, frustum_planes):
        for plane_points in frustum_planes:
            frustum_polygon = Poly3DCollection([plane_points], edgecolor='r', linewidths=1, alpha=0.2)
            ax.add_collection3d(frustum_polygon)

if __name__ == '__main__':
    frustumFilter = FrustumFilter()
    frustumFilter.init()
    # Initial observer
    observer_position = np.array([-5.0, 2.0, 0.0], dtype=np.float32)
    observer_direction = np.array([0.5, 0.5, 0])  # Assuming looking along the positive x-axis
    fov = 60.0  # Field of view in degrees
    aspect_ratio = 4/3  # Width/height ratio of the viewport
    near = 0.1
    far = 30.0

    frustum_planes = frustumFilter.compute_frustum(observer_position, observer_direction, fov, near, far)
    ####################################
    points_3d = np.random.rand(10000000, 3) * 20.0 - 10.0
    # points_3d = np.random.rand(1100000, 3) * 20.0 - 10.0
    # points_3d = np.array([
    #     [4.0, 0.0, 1.0],
    #     [7.0, 0.0, 1.0],
    #     [10.0, 0.0, 1.0],
    #     [-5.0, 0.0, 1.0],
    # ]).astype(np.float32)
    ####################################
    start = time.time()
    filtered_indices_opencl = frustumFilter.filter_points_by_frustum_opencl(points_3d, frustum_planes)
    passed_points = points_3d[filtered_indices_opencl == 1]

    print(len(passed_points))
    print("dt: %s" % ((time.time() - start)))

    failed_points = points_3d[filtered_indices_opencl == 0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    frustumFilter.plot_frustum(ax, frustum_planes)
    if len(passed_points) > 10000:
        passed_points = passed_points[:10000]
    # ax.scatter(failed_points[:, 0], failed_points[:, 1], failed_points[:, 2], color='gray', alpha=0.2)
    ax.scatter(passed_points[:, 0], passed_points[:, 1], passed_points[:, 2], color='green', alpha=0.4)
    plt.title('Scatter Plot of Points with Results')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    # Show the plot
    plt.show()