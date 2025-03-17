import sys
import open3d as o3d

# Load the PCD file
pcd = o3d.io.read_point_cloud(sys.argv[1])

# Print information about the point cloud
print(pcd)
print("Number of points:", len(pcd.points))

for p in pcd.points[:10]:
    print(p)
for p in pcd.points[-10:]:
    print(p)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="PCD Visualization")