import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

def plot_line_from_sensor(fig, ax, sensor_pos, look_direction, length, fov):
    azimuth, elevation = look_direction
    horizontal_fov, vertical_fov = fov

    ### look direction
    # Calculate the end point of the line based on azimuth, elevation, and length
    x_end = sensor_pos[0] + length * np.cos(azimuth) * np.cos(elevation)
    y_end = sensor_pos[1] + length * np.sin(azimuth) * np.cos(elevation)
    z_end = sensor_pos[2] + length * np.sin(elevation)
    # Plot the line from sensor position to the calculated end point
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end], color='blue', label='Look Direction')
    ### fov
    # Calculate the end point of the line based on fov + ength
    x_end1 = sensor_pos[0] + length * np.cos(azimuth - horizontal_fov/2.0) * np.cos(elevation)
    y_end1 = sensor_pos[1] + length * np.sin(azimuth - horizontal_fov/2.0) * np.cos(elevation)
    x_end2 = sensor_pos[0] + length * np.cos(azimuth + horizontal_fov/2.0) * np.cos(elevation)
    y_end2 = sensor_pos[1] + length * np.sin(azimuth + horizontal_fov/2.0) * np.cos(elevation)
    z_end = sensor_pos[2] + length * np.sin(elevation)
    ax.plot([sensor_pos[0], x_end1], [sensor_pos[1], y_end1], [sensor_pos[2], z_end], color='red', label='fovh')
    ax.plot([sensor_pos[0], x_end2], [sensor_pos[1], y_end2], [sensor_pos[2], z_end], color='red')
    # Calculate the end point of the line based on fov + ength
    x_end = sensor_pos[0] + length * np.cos(azimuth) * np.cos(np.radians(elevation))
    y_end = sensor_pos[1] + length * np.sin(azimuth) * np.cos(np.radians(elevation))
    z_end1 = sensor_pos[2] + length * np.sin(elevation - vertical_fov/2.0 )
    z_end2 = sensor_pos[2] + length * np.sin(elevation + vertical_fov/2.0 )
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end1], color='orange', label='fovv')
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end2], color='orange')
    # Plot the sensor position
    ax.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2], color='blue', label='Sensor')


def points_inside_fov(sensor_pos, sensor_look_dir, points, max_distance_xy, fov):
    horizontal_fov, vertical_fov = fov
    fov_points_inside = []
    fov_points_outside = []
    # Calculate sensor orientation
    sensor_azimuth, sensor_elevation = sensor_look_dir
    for point in points:
        # Calculate angles between sensor and point
        delta_x = point[0] - sensor_pos[0]
        delta_y = point[1] - sensor_pos[1]
        delta_z = point[2] - sensor_pos[2]
        distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
        distance = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        is_inside_fov = False
        if distance_xy < max_distance_xy:
            azimuth = np.arctan2(delta_y, delta_x)
            elevation = np.arctan2(delta_z, distance_xy)
            # Calculate angles between sensor and point
            angle_diff_azimuth = azimuth - sensor_azimuth
            angle_diff_elevation = elevation - sensor_elevation
            # Check if point is inside the FOV considering sensor orientation
            if (
                -horizontal_fov / 2 <= angle_diff_azimuth <= horizontal_fov / 2 and
                -vertical_fov / 2 <= angle_diff_elevation <= vertical_fov / 2
            ):
                is_inside_fov = True
        if is_inside_fov:
            fov_points_inside.append(point)
        else:
            fov_points_outside.append(point)
    return np.array(fov_points_inside), np.array(fov_points_outside)


# Function to plot the sensor, multiple points, and the FOV in 3D space
def plot_sensor_points_fov(fig, ax, sensor_pos, points, max_distance_xy, fov):
    # inside, outside = points_inside_fov(sensor_pos, points, horizontal_fov, vertical_fov)
    start = time.time()
    inside, outside = points_inside_fov(sensor_pos, sensor_look_dir, points, max_distance_xy, fov)
    print(start - time.time())
    # Plot the points
    ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2], color='green', label='Inside', s=8.0)
    ax.scatter(outside[:, 0], outside[:, 1], outside[:, 2], color='red', label='Outside', s=1.0)

##################################################################################################################################

# Generate 1000 random 3D points
np.random.seed(42)
num_points = 10000
points = np.random.rand(num_points, 3) * 2 * 10 -10

# Example sensor and points positions
sensor_pos = np.array([-3, -3, 0])  # Sensor position (x, y, z)
sensor_look_dir = (np.radians(45), np.radians(0))  # Sensor look direction (azimuth, elevation)
# sensor_look_dir = (np.radians(0), np.radians(90))  # Sensor look direction (azimuth, elevation)
# Sensor FOV angles in degrees
fov = (np.radians(120), np.radians(90))  # Horizontal FOV angle in degrees, Vertical FOV angle in degrees
max_distance_xy = 10.0


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sensor, points, and FOV
plot_sensor_points_fov(fig, ax, sensor_pos, points, max_distance_xy, fov)
plot_line_from_sensor(fig, ax, sensor_pos, sensor_look_dir, max_distance_xy, fov)

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Show legend
ax.legend()
# Show the plot
plt.show()