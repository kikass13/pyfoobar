import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

### point every 20 degree of sin
stepAngleDeg = 200.0
### arbitrary distance to angle convertion
### 2pi = 360° ~ 6m
distance = 24.0
distanceInPi = distance/2.0
maxAngleDeg = distanceInPi * 360.0
print(maxAngleDeg)
# Create XYZ points as described in the previous answer
# angles = np.arange(0, 360, 20)
angles = np.arange(0, maxAngleDeg, stepAngleDeg)
angles_rad = np.deg2rad(angles)
# x = angles_rad
# y = np.sin(angles_rad)
# z = np.cos(angles_rad)
x = angles_rad
y1 = 1.0 * np.sin(angles_rad/12.0)
y2 = -2.0 + 1.0 * np.sin(angles_rad/12.0)
z = np.zeros(len(angles_rad))
xyz_points1 = np.column_stack((x, y1, z))
xyz_points2 = np.column_stack((x, y2, z))

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_points1[:, 0], xyz_points1[:, 1], xyz_points1[:, 2])
ax.scatter(xyz_points2[:, 0], xyz_points2[:, 1], xyz_points2[:, 2])

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()