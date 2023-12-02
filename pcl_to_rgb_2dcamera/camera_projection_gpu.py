import numpy as np
import pyopencl as cl
import cv2
import time

from chunksSplit import split_array_indices
from coloredCube import create_colored_cube_array

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

# Define your OpenCL kernel code
kernel_code = """
__kernel void project_points(__constant float* points, __global float* projection, __constant float* extrinsic_matrix, __constant float* intrinsic_matrix) {
    int gid = get_global_id(0);
    
    // load homogenous point
    float4 point = vload4(gid, points);

    // Apply the transformation matrix
    float4 extrafo0 = vload4(0, extrinsic_matrix);
    float4 extrafo1 = vload4(1, extrinsic_matrix);
    float4 extrafo2 = vload4(2, extrinsic_matrix);
    float4 extrafo3 = vload4(3, extrinsic_matrix);
    float4 rotated_and_translated_point;
    rotated_and_translated_point.x = dot(point, extrafo0);
    rotated_and_translated_point.y = dot(point, extrafo1);
    rotated_and_translated_point.z = dot(point, extrafo2);
    rotated_and_translated_point.w = dot(point, extrafo3);

    float4 intrafo0 = vload4(0, intrinsic_matrix);
    float4 intrafo1 = vload4(1, intrinsic_matrix);
    float4 intrafo2 = vload4(2, intrinsic_matrix);
    float4 intrafo3 = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    float4 projected_point;
    projected_point.x = dot(rotated_and_translated_point, intrafo0);
    projected_point.y = dot(rotated_and_translated_point, intrafo1);
    projected_point.z = dot(rotated_and_translated_point, intrafo2);
    projected_point.w = dot(rotated_and_translated_point, intrafo3);

    // Perform perspective division
    if (projected_point.z != 0.0f) {
        projected_point.x /= projected_point.z;
        projected_point.y /= projected_point.z;
        projected_point.z = projected_point.z;
    }

    // Store the result in the projection array
    vstore4(projected_point, gid, projection);
}
"""

class CameraProjector:
    def __init__(self, debug=False):            
        # Create OpenCL context and queue
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.program = None
        self.dbg = debug
    def init(self):
        # Compile the OpenCL program
        self.program = cl.Program(self.context, kernel_code)
        self.program.build()
    def debug(self, msg):
        if self.dbg:
            print("Projector - %s" % msg)
    def project_points_to_camera_opencl(self, points_3d, colors, extrinsic_matrix, camera_matrix, h, w, filterPointsBehindCamera=True, sortPointsByZ=True): 
        if self.program.get_build_info(self.device, cl.program_build_info.STATUS) > 0:
            raise ValueError("Kernel not compiled, please run .init() method")
        begin = time.time()
        ### make points homogenous if not already
        if points_3d.shape[1] == 3:
            points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)))
        projection_result = np.empty_like(points_3d, dtype=np.float32)
        a,b,c = self.device.max_work_item_sizes
        N = a*b 
        chunks = int(np.floor(len(points_3d) / N) + 1)
        self.debug("processing %s points in %s chunks" % (len(points_3d), chunks))
        chunk_ranges = split_array_indices(len(points_3d), chunks)
        projection_result = np.empty_like(points_3d, dtype=np.float32)
        start = time.time()
        for start_index, end_index in chunk_ranges:
            points = points_3d[start_index:end_index].astype(np.float32)
            result = np.zeros_like(points, dtype=np.float32)
            # Create OpenCL buffers for the data
            points_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points.flatten().astype(np.float32))
            intrinsic_camera_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_matrix.flatten().astype(np.float32))
            extrinsic_camera_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=extrinsic_matrix.flatten().astype(np.float32))
            result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.size * result.itemsize)
            b1 = points_buffer.get_info(cl.mem_info.SIZE)
            b2 = intrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
            b3 = extrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
            b4 = result_buffer.get_info(cl.mem_info.SIZE)
            globalSize = next_power_of_2(b1+b2+b3+b4)
            localSize = 128
            # Execute the OpenCL kernel
            self.program.project_points(self.queue, (globalSize,), (localSize,), points_buffer, result_buffer, extrinsic_camera_buffer, intrinsic_camera_buffer)
            # Retrieve the result from the OpenCL buffer
            cl.enqueue_copy(self.queue, result, result_buffer).wait()
            projection_result[start_index:end_index] = result
        self.debug("ProjectionDt: %s" % (time.time() - start))
        ### filter points behind camera
        if filterPointsBehindCamera:
            start = time.time()
            indices = np.where(
                (projection_result[:,0] > w) | (projection_result[:,0] < 0)
                | (projection_result[:,1] > h) | (projection_result[:,1] < 0)
                | (projection_result[:,2] < 0))[0]
            projection_result = projection_result[indices]
            colors = colors[indices]
            self.debug("FilterDt: %s" % (time.time() - start))
        ### sort, so that near points are first
        if sortPointsByZ:
            start = time.time()
            z_values = projection_result[:, 2]
            sorted_indices = np.argsort(z_values)
            #sorted_indices = np.argsort(z_values)[::-1] ## descendng
            projection_result = projection_result[sorted_indices]
            colors = colors[sorted_indices]
            self.debug("SortDt: %s" % (time.time() - start))
        ### done
        self.debug("cycle: %s" % (time.time() - begin))
        return projection_result, colors

if __name__ == '__main__':
    observer_position = np.array([0, -5.0, 0], dtype=np.float32)
    # observer_direction = np.array([1.0, .0, 0]).astype(np.float32)

    focal_length = 600
    image_width = 1200
    image_height = 900
    fx = focal_length  # Focal length in x-direction
    fy = focal_length  # Focal length in y-direction
    cx = image_width / 2.0  # X-coordinate of the principal point
    cy = image_height / 2.0  # Y-coordinate of the principal point

    # Define your camera matrix (example: perspective projection)
    camera_matrix = np.array([[fx, 0, cx,0],
                            [0, fy, cy,0],
                            [0, 0, 1,0]], dtype=np.float32)
    #################### TEST default 3d
    ### +x forward, +y up, +z left
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.float32) 
    
    from rotatcheck import generate_rotation_matrices
    for rotation_matrix in generate_rotation_matrices():
        ### create colored cube for reference
        ### Front > blue = X+
        ### Back > yellow = X-
        ### Right > purple = Y+
        ### Left > cyan = Y-
        ### Up > red = Z+
        ### down > green = Z-
        points_3d, colors = create_colored_cube_array(N=20, size=2.0)
        colors = (colors * 255).astype(np.uint8)
        print(rotation_matrix)
        # tvec = np.array([observer_position[0], observer_position[1], observer_position[2]])
        #################### MANGLE
        ### cv2 image coordinates
        ### z 180° flip
        # rotation_matrix = np.array([[-1, 0, 0],
        #                             [0, -1, 0],
        #                             [0, 0, 1]], dtype=np.float32)
        # ## x -90° >>> +x foward, +z up, +y right
        # rotation_matrix = np.array([[1, 0, 0],
        #                             [0, 0, 1],
        #                             [0, -1, 0]], dtype=np.float32)
        ###
        ### switch up z and y?
        tvec = np.array([observer_position[0], observer_position[2], observer_position[1]])
        # Convert the rotation matrix to a Rodrigues rotation vector
        ### after rotation, the translation for cv2 looks like this
        # tvec = np.array([observer_position[1], observer_position[2], -observer_position[0]])
        ####################
        ### homogenous roation and translation vectors before building extrinsic matrix
        ### we change the rotation matrix around 
        extrinsic_matrix = np.column_stack((rotation_matrix, tvec))
        extrinsic_matrix = np.vstack([extrinsic_matrix, [0,0,0,1]])
        ### do projection
        projector = CameraProjector()
        projector.init()

        print(len(points_3d))
        start = time.time()
        points_2d, colors = projector.project_points_to_camera_opencl(points_3d, colors, extrinsic_matrix, camera_matrix, image_height, image_width)
        print(time.time() - start)
        ### channels in cv2 is bgr, not rgb - so we switch these up
        # Convert RGB array to BGR array
        colors = colors[:, ::-1]
        # visualize the 2D points using OpenCV
        img = np.zeros((image_height, image_width, 3), dtype=np.uint8) 
        for point, color in zip(points_2d, colors):
            try:
                img = cv2.circle(img, (int(point[0]), int(point[1])), 1, color.tolist(), -1) 
            except:
                print(point)
                pass
        # Display the image
        cv2.imshow('Projected Points', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
