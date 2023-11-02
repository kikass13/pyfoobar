
import open3d as o3d
import sys
import os
import pandas as pd

import numpy as np

csv_file = sys.argv[1]
filename = os.path.splitext(csv_file)[0]

# # Read data from CSV file
# data = []
# print("1")
# with open(csv_file, 'r') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         print(row[:6])
#         data.append(row[:6])

# # Convert data to a NumPy array
# numpy_array = np.array(data, dtype=np.float)  # Convert to NumPy array and set data type
# print("2")
# del data

# Create a Pandas dataframe iterator to read the CSV in chunks
desired_columns = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue']  # Replace with your column names
chunk_size = 1000  # Adjust the chunk size as needed
csv_reader = pd.read_csv(csv_file, chunksize=chunk_size, usecols=desired_columns)

print("1")
# Initialize an empty list to store NumPy arrays
numpy_arrays = []

# Iterate through chunks and convert each to a NumPy array
for chunk in csv_reader:
    numpy_arrays.append(chunk.to_numpy())

# Concatenate the list of NumPy arrays
numpy_array = np.concatenate(numpy_arrays, axis=0)
print("2")
xyz = numpy_array[:, 0:3]  # XYZ coordinates
rgb = numpy_array[:, 3:6]  # RGB color values

print("3")
# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz)  # Set XYZ coordinates
point_cloud.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # Normalize RGB values to range [0, 1]

del xyz
del rgb

print("4")

# o3d.visualization.draw_geometries([point_cloud])

pcd_file = filename + ".pcd"

print(o3d.io.write_point_cloud(pcd_file, point_cloud, write_ascii=True, compressed=False, print_progress=True))

