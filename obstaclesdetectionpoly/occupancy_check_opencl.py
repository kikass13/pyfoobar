import time 
import pyopencl as cl
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

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


########################################################################################################
### example setup for footprint, waypoints and obstacles
# Example constants
footprint_length = 1.0
footprint_width = 0.5
# Example trajectory poses and obstacles
waypoints = np.array([(20.0, 1.0), (15.0, -2.0), (17.0, 12.0), (23.0, 20.0)])  # List of (x, y) waypoints
waypoints = interpolate_waypoints(waypoints, num_points=1000)
trajectory = posesFromPoints(waypoints)
Lx = 25  # Length in meters
Ly = 25   # Width in meters
Nx = 100  # Number of cells along the length
Ny = 100  # Number of cells along the width
vx, vy = calculate_voxel_size(Lx, Ly, Nx, Ny)
grid_map = create_grid_map(Lx, Ly, Nx, Ny).astype(np.uint8)
obstacles = findObstaclePolygons(grid_map, (vx,vy))
########################################################################################################
### obstacle flatten, so that each waypoint can check against each obstacle polygon (with n vertices)
# Flatten obstacles' vertices
obstacle_vertices = []
obstacle_sizes = []
for obstacle in obstacles:
    vertices = list(obstacle.exterior.coords)  # Get coordinates of polygon
    obstacle_sizes.append(len(vertices))
    for vertex in vertices:
        obstacle_vertices.append(vertex)
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
### build kernel
program = cl.Program(context, program_src).build()
### prepare input/output data memory to proper formats
trajectory_data = np.array(trajectory, dtype=np.float32)  # Shape (num_trajectory_poses, 3)
obstacle_data = np.array(obstacle_vertices, dtype=np.float32)  # Flattened list of vertices
obstacle_sizes_data = np.array(obstacle_sizes, dtype=np.int32)  # List of obstacle sizes
# Allocate space for results (num_trajectory_poses x num_obstacles)
footprint_coords = np.zeros((len(trajectory), len(obstacles), 8), dtype=np.float32)
results = np.zeros((len(trajectory), len(obstacles)), dtype=np.int32)
### use prepared data formats to create opencl buffers with corresponding sizes
trajectory_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=trajectory_data)
obstacle_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=obstacle_data)
obstacle_sizes_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=obstacle_sizes_data)
footprint_coords_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, footprint_coords.nbytes)
results_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results.nbytes)
########################################################################################################
### kernel execution and time measurement
n = 30
times = np.zeros((n))
for i in range(0,n):
    start = time.time()
    ### execute the kernel
    program.check_footprint_intersection(queue, (len(obstacles),), None,
        trajectory_buffer, np.int32(len(trajectory)),
        obstacle_buffer, obstacle_sizes_buffer, np.int32(len(obstacles)),
        np.float32(footprint_length), np.float32(footprint_width),
        footprint_coords_buffer, results_buffer)
    ### retrieve results
    cl.enqueue_copy(queue, footprint_coords, footprint_coords_buffer).wait()
    cl.enqueue_copy(queue, results, results_buffer).wait()
    # Print results: each row corresponds to a trajectory pose, each column to an obstacle
    # print(results)
    end = time.time()
    times[i] = end - start
print(f"opencl executions n: {n}")
print(f"opencl min dt: {np.min(times)}")
print(f"opencl max dt: {np.max(times)}")
print(f"opencl mean dt: {np.mean(times)}")
########################################################################################################
### data extraction and plotting
intersection_indices = []
unique_collision_indices = set()
unique_noncollision_footprints = set()
unique_collision_footprints = set()
# Iterate through the results to find all intersections
for waypoint_index in range(results.shape[0]):  # Loop through waypoints
    for obstacle_index in range(results.shape[1]):  # Loop through obstacles
        footprintCoords = footprint_coords[waypoint_index][obstacle_index]
        vertices = footprintCoords.reshape(4, 2)  # Ensure it's a 4x2 array for [x, y]
        polygon = Polygon(vertices)
        collision = results[waypoint_index][obstacle_index] == 1
        if collision:
            intersection_indices.append((waypoint_index, obstacle_index))
            unique_collision_indices.add(waypoint_index)
            unique_collision_footprints.add(polygon)
        else:
            unique_noncollision_footprints.add(polygon)
# Now `intersection_indices` contains the pairs of (waypoint_index, obstacle_index) where intersections occurred
# for waypoint_idx, obstacle_idx in intersection_indices:
#     print(f"waypoint {waypoint_idx} collides with obstacle {obstacle_idx}")
fig, ax = plt.subplots(1, 1)
ax = drawTrajectory(ax, trajectory_data[:,:2])
ax = drawPolygons(ax, obstacles)
for p in unique_noncollision_footprints:
    if p not in unique_collision_footprints:
        ax = drawPolygon(ax, p, "blue")
    else:
        ax = drawPolygon(ax, p, "violet")
ax = drawTrajectoryCollisions(ax, trajectory_data, unique_collision_indices)
plt.show()

