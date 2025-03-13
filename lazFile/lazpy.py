import sys
import os
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np

import open3d as o3d

def read_laz_file_points(path):
    las = laspy.read(path)
    # Extract X, Y, Z coordinates
    return np.vstack((las.x, las.y, las.z)).transpose()

VOXEL_FILTER_SIZE = 0.1
# Load the LAZ file
file_paths = sys.argv[1:]
all_files = []
for path in file_paths:
    if os.path.isdir(path):
        all_files.extend([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    else:
        all_files.append(path)

###
def task(f):
    print(f"Processing '{f}' ...")
    points = read_laz_file_points(f)
    print(f"Loaded {len(points)} points")
    # Convert to Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ### voxel filter
    downpcd = pcd.voxel_down_sample(voxel_size=VOXEL_FILTER_SIZE)
    print(f"Downsampled to {len(downpcd.points)} points")
    return downpcd.points

# Using ThreadPoolExecutor to manage multiple threads and get results
with ThreadPoolExecutor(max_workers=10) as executor:
    # Map the function to the input range and collect results
    results = list(executor.map(task, all_files))

all_points = np.vstack([result for result in results])

OUTFILE_PATH = "downsampled_laz.pcd"
print(f"Merging Points into '{OUTFILE_PATH}' ...")
### safe all points to single pcl file
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(all_points)
print(o3d.io.write_point_cloud("downsampled_laz.pcd", merged_pcd, write_ascii=True, compressed=False, print_progress=True))

# Visualize
o3d.visualization.draw_geometries([merged_pcd])
