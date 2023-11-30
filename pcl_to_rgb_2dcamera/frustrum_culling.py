

import numpy as np
import time

import numba as nb

import numpy as np
from numba import njit, prange

@njit
def perspective_projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / np.tan(0.5 * np.radians(fov))
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect_ratio
    m[1,1] = f
    m[2,2] = (far + near) / (near - far)
    m[2,3] = 2.0 * far * near / (near - far)
    m[3,2] = -1.0
    print(m)
    return m

@njit
def is_point_in_frustum(point, projection_matrix, camera_position):
    trasnlated_point = (point - camera_position).astype(np.float32)
    homogeneous_point = np.append(trasnlated_point, 1.0).astype(np.float32)
    eye_space_point = np.dot(projection_matrix, homogeneous_point)
    # Check if the point is inside the view frustum
    # print(homogeneous_point)
    # print(eye_space_point)
    # print(eye_space_point[0]/eye_space_point[3])
    # print(eye_space_point[1]/eye_space_point[3])
    # print(eye_space_point[2]/eye_space_point[3])
    return -1 <= eye_space_point[0]/eye_space_point[3] <= 1 and \
           -1 <= eye_space_point[1]/eye_space_point[3] <= 1 and \
           -1 <= eye_space_point[2]/eye_space_point[3] <= 1

@njit
def filter_points_by_frustum_numba(points_3d, camera_position, fov, aspect_ratio, near, far):
    num_points = len(points_3d)
    projection_matrix = perspective_projection_matrix(fov, aspect_ratio, near, far)
    result = np.zeros(num_points, dtype=np.bool_)
    for i in prange(num_points):
        result[i] = is_point_in_frustum(points_3d[i], projection_matrix, camera_position)
    filtered_points = points_3d[result]
    return filtered_points

# Example usage
camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
fov = 60.0  # Field of view in degrees
aspect_ratio = 16/9  # Width/height ratio of the viewport
near = 0.1
far = 100.0

# example 1M points
points_3d = np.random.rand(1000000, 3).astype(np.float32) * 20.0 - 10.0

points_3d[10000] = np.array([5.0,1.0,1.0])

print(len(points_3d))
start = time.time()
filtered_points_numba = filter_points_by_frustum_numba(points_3d, camera_position, fov, aspect_ratio, near, far)
print(len(filtered_points_numba))
print("dt1: %s" % (time.time() - start))


