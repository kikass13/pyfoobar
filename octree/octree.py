
import pcl
import sys
import time 

t = time.time()
# Load the PCD file
pcd_file = sys.argv[1]
cloud = pcl.load(pcd_file)
print(f"Loading map dt: {time.time() - t}")

t = time.time()
# Define octree resolution
resolution = 0.1  # Adjust based on the scale of your point cloud
octree = cloud.make_octreeSearch(resolution) 
octree.add_points_from_input_cloud()
print(f"Generating octree dt: {time.time() - t}")

# Define search parameters
search_point = cloud[1000]  # Example: using an existing point in the cloud
search_radius = 100.0  # Radius within which to search
# Perform the radius search
t = time.time()
[indices, sqr_distances] = octree.radius_search(search_point, search_radius)
print(f"Radius search dt: {time.time() - t}")

# Print the results
print(f"Found {len(indices)} points within {search_radius}m of {search_point}")
# for i, idx in enumerate(indices):
#     print(f"Point {idx}: {cloud[idx]} (Distance: {sqr_distances[i]})")