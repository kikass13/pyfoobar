import time 
import re
import pyopencl as cl
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numba as nb

from occupancy_trajectory_check import drawTrajectory, interpolate_waypoints, posesFromPoints
from findcontours import drawPolygons, create_grid_map, findObstaclePolygons, calculate_voxel_size

def drawPolygon(ax, polygon, color):
    x, y = polygon.exterior.xy  # Get the exterior coordinates of the single polygon
    ax.fill(x, y, color=f"dark{color}", alpha=1.0, label='Combined Polygon')
    ax.plot(x, y, color=color, linewidth=1)  # Draw the boundary
    return ax

def drawTrajectoryCollisions(ax, trajectory, unique_collision_indices):
    # Plot the trajectory and mark each point with its status
    for traj_index in unique_collision_indices:
        x,y,theta = trajectory[traj_index]
        ax.plot(x, y, 'o', color="violet")
    return ax


MAX_OBSTACLE_VERTICES = 10

def preprocess_obstacles(obstacles, max_vertices):
    """Convert Shapely polygons to padded NumPy arrays."""
    processed_obstacles = np.zeros((len(obstacles), 1 + max_vertices * 2))
    for i, obstacle in enumerate(obstacles):
        vertices = np.array(obstacle.exterior.coords)[:-1]  # Remove closing point
        num_vertices = len(vertices)
        vertices_flat = vertices.flatten()
        # copy num vertices & copy vertices
        processed_obstacles[i, 0] = num_vertices
        processed_obstacles[i, 1:len(vertices_flat)+1] = vertices_flat
    return processed_obstacles

@nb.njit()
def prepare_combined_data_optimized(trajectory, processed_obstacles):
    """
    Generate the combined data for the kernel using preprocessed obstacles.
    """
    entry_size = 3 + processed_obstacles.shape[1]  # (x, y, theta) + num_vertices + flattened vertices
    num_pairs = len(trajectory) * len(processed_obstacles)
    combined_data = np.zeros((num_pairs, entry_size), dtype=np.float32)
    idx = 0
    for i in range(len(trajectory)):
        x, y, theta = trajectory[i]
        for obstacle in processed_obstacles:
            # vertices, num_vertices = obstacle  # Extract obstacle data
            num_vertices = obstacle[0]
            vertices = obstacle[1:]
            # Write to the array
            combined_data[idx, 0] = x  # Trajectory x
            combined_data[idx, 1] = y  # Trajectory y
            combined_data[idx, 2] = theta  # Trajectory theta
            combined_data[idx, 3] = num_vertices  # Number of vertices
            combined_data[idx, 4:] = vertices
            idx += 1
    return entry_size, combined_data
prepare_combined_data_optimized(np.zeros((1000,3), dtype=np.float32), np.zeros((200, 1+MAX_OBSTACLE_VERTICES*2))) ### numba precompile

### despite this function using numpy broadcast funtions, it is slower by a factor of 2 to 4 than the numba version
# def prepare_combined_data_vectorized(trajectory, processed_obstacles):
#     """
#     Vectorized generation of combined data for the kernel.
#     """
#     num_trajectory = trajectory.shape[0]
#     num_obstacles = processed_obstacles.shape[0]
#     ### Calculate the entry size: (x, y, theta) + num_vertices + max_vertices * 2
#     entry_size = 3 + processed_obstacles.shape[1]
#     ### Preallocate combined data array
#     combined_data = np.zeros((num_trajectory * num_obstacles, entry_size), dtype=np.float32)
#     ### Create trajectory-obstacle pairs using broadcasting
#     trajectory_repeated = np.repeat(trajectory, num_obstacles, axis=0)
#     obstacles_tiled = np.tile(processed_obstacles, (num_trajectory, 1))
#     ### Fill combined_data
#     combined_data[:, :3] = trajectory_repeated  # x, y, theta
#     combined_data[:, 3:] = obstacles_tiled[:]  # num_vertices
#     return entry_size, combined_data

########################################################################################################
### example setup for footprint, waypoints and obstacles
# Example constants
footprint_length = 4.0
footprint_width = 2.0
# Example trajectory poses and obstacles
waypoints = np.array([(20.0, 1.0), (15.0, -2.0), (17.0, 12.0), (23.0, 20.0)])  # List of (x, y) waypoints
waypoints = interpolate_waypoints(waypoints, num_points=300)
trajectory = posesFromPoints(waypoints)
Lx = 25  # Length in meters
Ly = 25   # Width in meters
Nx = 100  # Number of cells along the length
Ny = 100  # Number of cells along the width
vx, vy = calculate_voxel_size(Lx, Ly, Nx, Ny)
grid_map = create_grid_map(Lx, Ly, Nx, Ny).astype(np.uint8)
obstacles = findObstaclePolygons(grid_map, (vx,vy))
### duplicate obstacles for testing purposes
obstacles *= 100
print(f"trajectory points: {len(trajectory)}")
print(f"obstacles count: {len(obstacles)}")
########################################################################################################
for i in range(0,2):
    t0 = time.time()
    # Combine trajectory data and obstacle data (each trajectory-obstacle pair)
    num_trajectory_poses = len(trajectory)
    num_obstacles = len(obstacles)
    ##########
    t1 = time.time()
    prepared_obstacles = preprocess_obstacles(obstacles, MAX_OBSTACLE_VERTICES)
    t1end = time.time() - t1
    t2 = time.time()
    # COMBINED_DATA_LENGTH, combined_data = prepare_combined_data_vectorized(trajectory, prepared_obstacles) ### slower
    COMBINED_DATA_LENGTH, combined_data = prepare_combined_data_optimized(trajectory, prepared_obstacles)
    t2end = time.time() - t2
    t3 = time.time()
    num_pairs = len(combined_data)
    footprint_coords = np.zeros((num_pairs, 8), dtype=np.float32)
    results = np.zeros((num_pairs,), dtype=np.int32)
    t3end = time.time() - t3
    if i > 0:
        print(f"data preprocess obstacles dt: {t1end}")
        print(f"data prepae combined data dt: {t2end}")
        print(f"data malloc result buffers : {t3end}")
        print(f"data prep sum dt: {time.time() - t0}")
########################################################################################################
### opencl context setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
########################################################################################################
### kernel compilation and data buffer preparation
### load and compile OpenCL program
with open('kernel.cl', 'r') as f:
    program_src = f.read()
    pattern = r'#define MAX_OBSTACLE_VERTICES \d+'
    program_src = re.sub(pattern, f"#define MAX_OBSTACLE_VERTICES {MAX_OBSTACLE_VERTICES}", program_src)
    pattern = r'#define COMBINED_DATA_LENGTH \d+'
    program_src = re.sub(pattern, f"#define COMBINED_DATA_LENGTH {COMBINED_DATA_LENGTH}", program_src)
# Iterate through the results to find all combined checks

### build kernel
program = cl.Program(context, program_src).build()
### use prepared data formats to create opencl buffers with corresponding sizes
combined_data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=combined_data)
footprint_coords_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, footprint_coords.nbytes)
result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, num_pairs * results.dtype.itemsize)
########################################################################################################
### kernel execution and time measurement
n = 30
times = np.zeros((n))
for i in range(0,n):
    start = time.time()
    ### execute the kernel
    program.check_footprint_collision(queue, (num_pairs,), None, 
        np.float32(footprint_length), np.float32(footprint_width),
        combined_data_buffer, np.int32(num_pairs),
        footprint_coords_buffer,
        result_buffer
    )
    ### retrieve results
    cl.enqueue_copy(queue, footprint_coords, footprint_coords_buffer).wait()
    cl.enqueue_copy(queue, results, result_buffer).wait()
    # Print results: each row corresponds to a trajectory pose, each column to an obstacle
    # print(results)
    end = time.time()
    times[i] = end - start
print(f"opencl executions n: {n}")
print(f"opencl min dt: {np.min(times)}")
print(f"opencl max dt: {np.max(times)}")
print(f"opencl mean dt: {np.mean(times)}")
########################################################################################################
### data extraction and plotting (dirty but okay for now)
intersection_indices = []
unique_collision_indices = set()
unique_noncollision_footprints = set()
unique_collision_footprints = set()
# Iterate through the results to find all combined checks
for i, entry in enumerate(combined_data):
    trajectory_id = int(i / num_obstacles)
    footprintCoords = footprint_coords[i]
    vertices = footprintCoords.reshape(4, 2)  # Ensure it's a 4x2 array for [x, y]
    polygon = Polygon(vertices)
    x, y, theta, num_vertices = entry[:4]
    collision = results[i]
    if collision == 1:
        unique_collision_indices.add(trajectory_id)
        unique_collision_footprints.add(polygon)
    else:
        unique_noncollision_footprints.add(polygon)
fig, ax = plt.subplots(1, 1)
ax = drawTrajectory(ax, trajectory[:,:2])
ax = drawPolygons(ax, obstacles)
for p in unique_noncollision_footprints:
    if p not in unique_collision_footprints:
        ax = drawPolygon(ax, p, "blue")
for p in unique_collision_footprints:
    ax = drawPolygon(ax, p, "violet")
ax = drawTrajectoryCollisions(ax, trajectory, unique_collision_indices)
plt.show()

