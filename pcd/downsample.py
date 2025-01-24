import open3d as o3d


file = "20250107-fbx-z-offset-118.5m.pcd"
voxel_size = 1.0 #m
outfile = f"downsample_{voxel_size}m_{file}"

# Load the PCD file
pcd = o3d.io.read_point_cloud(file)
print("Original point cloud:")
print(pcd)

# Visualize the original point cloud
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# Apply a voxel filter (down-sampling)
downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

print("Downsampled point cloud:")
print(downsampled_pcd)

# Visualize the downsampled point cloud
o3d.visualization.draw_geometries([downsampled_pcd], window_name="Downsampled Point Cloud")

# Save the downsampled point cloud to a new file
o3d.io.write_point_cloud(outfile, downsampled_pcd)
print(f"Downsampled point cloud saved as {outfile}")