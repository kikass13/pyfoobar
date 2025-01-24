import time

import open3d as o3d
import numpy as np

# Load the PCD file
pcd_file_path = "20250107-fbx-z-offset-118.5m.pcd"
t = time.time()
point_cloud = o3d.io.read_point_cloud(pcd_file_path)
print(f"{time.time() - t}s")
# Convert to NumPy array
points = np.asarray(point_cloud.points)

# Optional: If you need color or normals
colors = np.asarray(point_cloud.colors)    # For RGB data (if available)
normals = np.asarray(point_cloud.normals)  # For normals (if available)

# Display the result
print("Points shape:", points.shape)
print(points)

# Save the points to a .npy file (optional)
np.save("output_points.npy", points)
