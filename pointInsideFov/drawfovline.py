import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_line_from_sensor(sensor_pos, azimuth, elevation, length, horizontalfov, verticalfov):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ### look direction
    # Calculate the end point of the line based on azimuth, elevation, and length
    x_end = sensor_pos[0] + length * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
    y_end = sensor_pos[1] + length * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
    z_end = sensor_pos[2] + length * np.sin(np.radians(elevation))
    # Plot the line from sensor position to the calculated end point
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end], color='blue', label='Look Direction')



    ### fov
    # Calculate the end point of the line based on fov + ength
    x_end1 = sensor_pos[0] + length * np.cos(np.radians(azimuth - horizontalfov/2.0)) * np.cos(np.radians(elevation))
    y_end1 = sensor_pos[1] + length * np.sin(np.radians(azimuth - horizontalfov/2.0)) * np.cos(np.radians(elevation))
    x_end2 = sensor_pos[0] + length * np.cos(np.radians(azimuth + horizontalfov/2.0)) * np.cos(np.radians(elevation))
    y_end2 = sensor_pos[1] + length * np.sin(np.radians(azimuth + horizontalfov/2.0)) * np.cos(np.radians(elevation))
    z_end = sensor_pos[2] + length * np.sin(np.radians(elevation))
    ax.plot([sensor_pos[0], x_end1], [sensor_pos[1], y_end1], [sensor_pos[2], z_end], color='red', label='fovh')
    ax.plot([sensor_pos[0], x_end2], [sensor_pos[1], y_end2], [sensor_pos[2], z_end], color='red')

    # Calculate the end point of the line based on fov + ength
    x_end = sensor_pos[0] + length * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
    y_end = sensor_pos[1] + length * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
    z_end1 = sensor_pos[2] + length * np.sin(np.radians(elevation - verticalfov/2.0 ))
    z_end2 = sensor_pos[2] + length * np.sin(np.radians(elevation + verticalfov/2.0 ))
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end1], color='orange', label='fovv')
    ax.plot([sensor_pos[0], x_end], [sensor_pos[1], y_end], [sensor_pos[2], z_end2], color='orange')



    # Plot the sensor position
    ax.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2], color='blue', label='Sensor')

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

# Example sensor position (x, y, z), azimuth, elevation, and length
sensor_pos = np.array([-3, -3, 0])  # Sensor position (x, y, z)
azimuth = 45  # Azimuth angle in degrees
elevation = 0  # Elevation angle in degrees
length = 10  # Length of the line

horizontalfov = 120
verticalfov = 90

# Plot the line from sensor position with the given azimuth, elevation, and length
plot_line_from_sensor(sensor_pos, azimuth, elevation, length, horizontalfov, verticalfov)
