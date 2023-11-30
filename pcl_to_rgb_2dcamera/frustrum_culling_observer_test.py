import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_frustum(observer_position, observer_direction, fov_degrees, near_clip, far_clip, aspect_ratio=1.0):
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
    return (near_plane, far_plane, top_plane, right_plane, bottom_plane, left_plane)

def is_point_in_frustum(point, frustum_planes):
    def compute_normal_vector(plane_points):
        line1 = plane_points[1] - plane_points[0]
        line2 = plane_points[2] - plane_points[0]
        normal = np.cross(line1, line2)
        ### normalize normal
        normal /= np.linalg.norm(normal)
        # print(normal)
        return normal
    for plane_points in frustum_planes:
        normal_vector = compute_normal_vector(plane_points)
        # Choose any point from the plane to calculate the distance
        reference_point = plane_points[0]
        distance = -np.dot(normal_vector, reference_point)
        vector_to_point = point - reference_point
        # print(distance)
        # Check against the frustum plane
        dotP = np.dot(vector_to_point, normal_vector)
        if dotP < 0:
            return False
    return True


def plot_frustum(ax, frustum_planes):
    for plane_points in frustum_planes:
        frustum_polygon = Poly3DCollection([plane_points], edgecolor='r', linewidths=1, alpha=0.2)
        ax.add_collection3d(frustum_polygon)
   
# Initial observer
observer_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
observer_direction = np.array([1, 0, 0])  # Assuming looking along the positive x-axis

fov = 60.0  # Field of view in degrees
aspect_ratio = 4/3  # Width/height ratio of the viewport
near = 0.1
far = 9.0

# example 1M points
points_3d = np.random.rand(1000, 3).astype(np.float32) * 20.0 - 10.0
# points_3d = np.array([
#     [4.0, 0.0, 1.0],
#     [7.0, 0.0, 1.0],
#     [10.0, 0.0, 1.0],
#     [-5.0, 0.0, 1.0],
# ]).astype(np.float32)
# Initial computation of frustum planes
frustum_planes = compute_frustum(observer_position, observer_direction, fov, near, far)
# Change in observer direction
new_observer_direction = np.array([1, 0, 0])  # Example: now looking along the positive x-axis

# Check if the point is inside the updated frustum
result = np.zeros(len(points_3d), dtype=np.bool_)
for i,p in enumerate(points_3d):
    result[i] = is_point_in_frustum(p, frustum_planes)

passed = points_3d[result == 1]
failed = points_3d[result == 0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plot_frustum(ax, frustum_planes)

ax.scatter(failed[:, 0], failed[:, 1], failed[:, 2], color='gray', alpha=0.2)
ax.scatter(passed[:, 0], passed[:, 1], passed[:, 2], color='green')

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
