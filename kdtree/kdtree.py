import numpy as np
from scipy.spatial import KDTree
import time

# Example data: Sampled points along road reference lines
# Each entry is a tuple of (x, y) coordinates and a road ID for reference
# In practice, you'd replace these with actual coordinates from your OpenDRIVE map.
road_points = [
    (100.0, 200.0, 'road_1'),  # Sample point on road_1
    (102.0, 198.5, 'road_1'),
    (300.0, 400.0, 'road_2'),  # Sample point on road_2
    (305.0, 405.0, 'road_2'),
    # Add more points as needed for other roads...
]

# Separate the coordinates and road IDs for k-d tree construction
coordinates = np.array([(x, y) for x, y, _ in road_points])  # (x, y) coordinates only
road_ids = [road_id for _, _, road_id in road_points]        # List of road IDs


# Step 2: Build the k-d tree using the coordinates
start = time.time()
kd_tree = KDTree(coordinates)
print(f"Time for creating tree: {(time.time() - start) * 1000.0 :.3f} ms")

# Function to find the nearest road given the robot's current position
def find_nearest_road(robot_position):
    # Step 3: Query the k-d tree for the nearest neighbor to the robot's position
    distance, index = kd_tree.query(robot_position)  # Returns the distance and index of nearest point
    
    # Get the nearest road ID from the original data
    nearest_road_id = road_ids[index]
    
    # Return the result
    return {
        'road_id': nearest_road_id,
        'nearest_point': coordinates[index],
        'distance': distance
    }

# Example usage
robot_position = (101.0, 199.0)  # Current position of the robot
start = time.time()
nearest_road_info = find_nearest_road(robot_position)
print(f"Time for finding dist in tree: {(time.time() - start) * 1000.0 :.3f} ms")

# Output the nearest road information
print("Nearest road ID:", nearest_road_info['road_id'])
print("Nearest point on the road:", nearest_road_info['nearest_point'])
print("Distance to nearest point:", nearest_road_info['distance'])