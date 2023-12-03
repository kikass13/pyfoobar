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
void drawPixel(__global uchar* projection, uint index, uint u, uint v, half pixeldepth, uint image_width, uint image_height,
               __constant uchar* colors, __global half* depths)
{
    /// check if pixel behind camera && inside image boundaries
    if(pixeldepth < 0 &&u > 0 && u < image_width && v > 0 && v < image_height)
    {
        uint i = v * image_width + u;
        half d = depths[i];
        if(pixeldepth < d){
            return;
        }
        // overwrite depth
        depths[i] = pixeldepth;
        // load color of pixel from point
        uchar3 rgb = vload3(index, colors);
        vstore3(rgb, i, projection);
    }
}

__kernel void project_points(uint offset, __constant float* points, __constant uchar* colors, __global half* depths, 
                            __global uchar* projection, __constant float* extrinsic_matrix, __constant float* intrinsic_matrix,
                            uchar inflation) 
{
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

    // get image properties
    uint image_width = intrafo0.z * 2.0f;
    uint image_height = intrafo1.z * 2.0f;

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

    // get pixel depth of the calculated pixel
    // if our depth is greater then that existing depth, we overwrite it
    /// we round the pixel coordinates to nearest integer
    uint u = floor(projected_point.x + 0.5f);
    uint v = floor(projected_point.y + 0.5f);

    uint index = gid + offset;

    // Store the pixel in the projection image
    if(inflation == 1){
        drawPixel(projection, index, u, v, projected_point.z, image_width, image_height, colors, depths);
    }
    // draw rect around pixel 
    else{
        uint startX = u - (uint)inflation / 2.0f;
        uint startY = v - (uint)inflation / 2.0f;
        uint endX =  u + (uint)inflation / 2.0f;
        uint endY =  v + (uint)inflation / 2.0f;
        for (int y = startY; y <= endY; ++y) {
            uint tmp = y * image_width;
            for (int x = startX; x <= endX; ++x) {
                drawPixel(projection, index, x, y, projected_point.z, image_width, image_height, colors, depths);
            }
        }
    }
    // draw cross
    // drawPixel(projection, index, u, v, projected_point.z, image_width, image_height, colors, depths);
    // drawPixel(projection, index, u-1, v, projected_point.z, image_width, image_height, colors, depths);
    // drawPixel(projection, index, u+1, v, projected_point.z, image_width, image_height, colors, depths);
    // drawPixel(projection, index, u, v-1, projected_point.z, image_width, image_height, colors, depths);
    // drawPixel(projection, index, u, v+1, projected_point.z, image_width, image_height, colors, depths);
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
    def project_points_to_camera_opencl(self, points_3d, colors, extrinsic_matrix, camera_matrix, inflation=1): 
        if self.program.get_build_info(self.device, cl.program_build_info.STATUS) > 0:
            raise ValueError("Kernel not compiled, please run .init() method")
        begin = time.time()
        ### make points homogenous if not already
        if points_3d.shape[1] == 3:
            points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)))
        ### get image props
        image_width = int(camera_matrix[0,2] * 2.0)
        image_height = int(camera_matrix[1,2] * 2.0)
        ### create outputs
        projection_result = np.zeros((image_height, image_width, 3), dtype=np.uint8) 
        ### return if no points
        if len(points_3d) == 0:
            return projection_result
        ### chunks
        a,b,c = self.device.max_work_item_sizes
        N = a*b
        chunks = int(np.floor(len(points_3d) / N) + 1)
        self.debug("processing %s points in %s chunks" % (len(points_3d), chunks))
        projection_depths = np.full((image_height, image_width), -99.999, dtype=np.float16) 
        chunk_ranges = split_array_indices(len(points_3d), chunks)
        colors_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=colors.flatten())
        intrinsic_camera_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=camera_matrix.flatten().astype(np.float32))
        extrinsic_camera_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=extrinsic_matrix.flatten().astype(np.float32))
        result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, projection_result.size * projection_result.itemsize)
        start = time.time()
        for chunk_counter, indices in enumerate(chunk_ranges):
            start_index, end_index = indices
            points = points_3d[start_index:end_index].astype(np.float32)
            # Create OpenCL buffers for the data
            points_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points.flatten())
            depth_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=projection_depths.flatten())
            b1 = points_buffer.get_info(cl.mem_info.SIZE)
            b2 = intrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
            b3 = extrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
            b4 = result_buffer.get_info(cl.mem_info.SIZE)
            globalSize = next_power_of_2(b1+b2+b3+b4)
            localSize = 128
            # Execute the OpenCL kernel
            self.program.project_points(self.queue, (globalSize,), (localSize,), np.uint32(start_index), points_buffer, colors_buffer, depth_buffer, 
                                        result_buffer, extrinsic_camera_buffer, intrinsic_camera_buffer, 
                                        np.uint8(inflation)
            )
            # Retrieve the result from the OpenCL buffer
            cl.enqueue_copy(self.queue, projection_result, result_buffer).wait()
            ### if this is not the last chunk
            if chunk_counter+1 < len(chunk_ranges):
                ### copy back the depth buffer, for the next chunk iteration
                cl.enqueue_copy(self.queue, projection_depths, depth_buffer).wait()
        self.debug("ProjectionDt: %s" % (time.time() - start))
        # return cv2.flip(projection_result, 1)
        return projection_result

if __name__ == '__main__':
    observer_position = np.array([0, -4.0, 0], dtype=np.float32)
    # observer_direction = np.array([1.0, .0, 0]).astype(np.float32)

    focal_length = 600
    image_width = 1200
    image_height = 900
    fx = focal_length  # Focal length in x-direction
    fy = focal_length  # Focal length in y-direction
    cx = image_width / 2.0  # X-coordinate of the principal point
    cy = image_height / 2.0  # Y-coordinate of the principal point

    # Define your camera matrix
    camera_matrix = np.array([
        [fx, 0, cx,0],
        [0, fy, cy,0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
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
        points_3d, colors = create_colored_cube_array(N=100, size=2.0)
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
        img = projector.project_points_to_camera_opencl(points_3d, colors, extrinsic_matrix, camera_matrix, inflation=1)
        print(time.time() - start)
        ### channels in cv2 is bgr, not rgb - so we switch these up
        # Convert RGB array to BGR array
        img = img[:, :, ::-1]
        # Display the image
        cv2.imshow('Projected Points', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
