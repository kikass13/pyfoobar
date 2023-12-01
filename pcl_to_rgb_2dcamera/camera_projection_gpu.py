import numpy as np
import pyopencl as cl
import cv2
import sys

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
__kernel void project_points(__global float* points, __global float* projection, __global float* extrinsic_matrix, __global float* intrinsic_matrix) {
    int gid = get_global_id(0);
    
    // if(gid > 100) return;

    float3 point;
    point[0] = points[gid + 0];
    point[1] = points[gid + 1];
    point[2] = points[gid + 2];

    // printf("[%u]p: %f, %f, %f\\n", gid,point[0],point[1],point[2]);
    // printf("[%u]E: %f, %f, %f\\n", gid,extrinsic_matrix[0*4 + 3],extrinsic_matrix[1*4 + 3],extrinsic_matrix[2*4 + 3]);
    
    float3 rotated_and_translated_point;
    // Apply extrinsic matrix (rotation and translation)             r1                                r2                                     r3                                 txyz                               
    rotated_and_translated_point[0] = extrinsic_matrix[0*4 + 0] * point[0] + extrinsic_matrix[0*4 + 1] * point[1] + extrinsic_matrix[0*4 + 2] * point[2] + extrinsic_matrix[0*4 + 3];
    rotated_and_translated_point[1] = extrinsic_matrix[1*4 + 0] * point[0] + extrinsic_matrix[1*4 + 1] * point[1] + extrinsic_matrix[1*4 + 2] * point[2] + extrinsic_matrix[1*4 + 3];
    rotated_and_translated_point[2] = extrinsic_matrix[2*4 + 0] * point[0] + extrinsic_matrix[2*4 + 1] * point[1] + extrinsic_matrix[2*4 + 2] * point[2] + extrinsic_matrix[2*4 + 3];

    // printf("[%u]P': %f, %f, %f\\n", gid,rotated_and_translated_point[0],rotated_and_translated_point[1],rotated_and_translated_point[2]);
    // printf("[%u]I: %f, %f, %f\\n", gid,intrinsic_matrix[0*4 + 0],intrinsic_matrix[1*4 + 1],intrinsic_matrix[2*4 + 2]);

    // Apply intrinsic matrix (perspective projection)       x                                      y                                                           z
    float x = intrinsic_matrix[0*4 + 0] * rotated_and_translated_point[0] + intrinsic_matrix[0*4 + 1] * rotated_and_translated_point[1] + intrinsic_matrix[0*4 + 2] * rotated_and_translated_point[2];
    float y = intrinsic_matrix[1*4 + 0] * rotated_and_translated_point[0] + intrinsic_matrix[1*4 + 1] * rotated_and_translated_point[1] + intrinsic_matrix[1*4 + 2] * rotated_and_translated_point[2];
    float w = intrinsic_matrix[2*4 + 0] * rotated_and_translated_point[0] + intrinsic_matrix[2*4 + 1] * rotated_and_translated_point[1] + intrinsic_matrix[2*4 + 2] * rotated_and_translated_point[2];
    
	// safety for w
	//w = (float) w + 0.00001f;

    // printf("[%u]XYW: %f, %f, %f\\n", gid,x,y,w);
	
    // Perspective division
    float inv_w = 1.0f / w;

    // Store the result in the projection array
    projection[gid + 0] = x * inv_w;
    projection[gid + 1] = y * inv_w;
    projection[gid + 2] = w;

    // printf("[%u]L: %f, %f\\n", gid, projection[gid + 0], projection[gid + 1]);

}
"""

def project_points_to_camera_opencl(points_3d, colors, extrinsic_matrix, camera_matrix):
  
    # Create OpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Compile the OpenCL program
    program = cl.Program(context, kernel_code).build()
    projection_result = np.empty_like(points_3d, dtype=np.float32)

    # Create OpenCL buffers for the data
    points_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(points_3d).flatten().astype(np.float32))
    intrinsic_camera_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(camera_matrix).flatten().astype(np.float32))
    extrinsic_camera_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(extrinsic_matrix).flatten().astype(np.float32))
    projection_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, projection_result.size * projection_result.itemsize)

    b1 = points_buffer.get_info(cl.mem_info.SIZE)
    b2 = intrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
    b3 = extrinsic_camera_buffer.get_info(cl.mem_info.SIZE)
    b4 = projection_buffer.get_info(cl.mem_info.SIZE)
    globalSize = next_power_of_2(b1+b2+b3)
    localSize = 128
    # Execute the OpenCL kernel
    program.project_points(queue, (globalSize,), (localSize,), points_buffer, projection_buffer, extrinsic_camera_buffer, intrinsic_camera_buffer)

    # Retrieve the result from the OpenCL buffer
    cl.enqueue_copy(queue, projection_result, projection_buffer).wait()
    return projection_result



if __name__ == '__main__':
    N = 50
    # points_3d = np.random.rand(N, 3).astype(np.float32) * 20.0 - 10
    # colors = (np.random.rand(N, 3) * 255.0).astype(np.uint8) 
    ### create colored cube for reference
    ### Front > blue = X+
    ### Back > yellow = X-
    ### Right > purple = Y+
    ### Left > cyan = Y-
    ### Up > red = Z+
    ### down > green = Z-
    points_3d, colors = create_colored_cube_array(N=N, size=2.0)
    colors = (colors * 255).astype(np.uint8)

    observer_position = np.array([-5.0, 2.0, 2.0], dtype=np.float32)
    # observer_direction = np.array([1.0, .0, 0]).astype(np.float32)

    focal_length = 600
    image_width = 1200
    image_height = 900
    fx = focal_length  # Focal length in x-direction
    fy = focal_length  # Focal length in y-direction
    cx = image_width / 2.0  # X-coordinate of the principal point
    cy = image_height / 2.0  # Y-coordinate of the principal point

    # Define your camera matrix (example: perspective projection)
    camera_matrix = np.array([[fx, 0, cx ],
                            [0, fy, cy ],
                            [0, 0, 1 ]], dtype=np.float32)
    #################### TEST default 3d
    ### +x forward, +y up, +z left
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.float32) 
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
    # rotation_matrices = [
    #     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    #     np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
    #     np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    #     np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
    #     np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
    #     np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    # ]
    ###
    ### switch up z and y?
    tvec = np.array([observer_position[0], observer_position[2], observer_position[1]])
    # Convert the rotation matrix to a Rodrigues rotation vector
    ### after rotation, the translation for cv2 looks like this
    # tvec = np.array([observer_position[1], observer_position[2], -observer_position[0]])
    ####################
    extrinsic_matrix = np.column_stack((rotation_matrix, tvec))
    ### do projection
    points_2d = project_points_to_camera_opencl(points_3d, colors, extrinsic_matrix, camera_matrix)
    ### channels in cv2 is bgr, not rgb - so we switch these up
    # Convert RGB array to BGR array
    colors = colors[:, ::-1]
    # visualize the 2D points using OpenCV
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8) 
    for point, color in zip(points_2d, colors):
        if point[2] > 0:
            continue
        try:
            img = cv2.circle(img, (int(point[0]), int(point[1])), 1, color.tolist(), -1) 
        except:
            # print(point)
            pass
    # Display the image
    cv2.imshow('Projected Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
