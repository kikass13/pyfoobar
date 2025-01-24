import time 
import numpy as np
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

from occupancy_trajectory_check import calculate_voxel_size, create_grid_map, interpolate_waypoints, posesFromPoints, drawTrajectory
from driving_corridor import create_robot_footprint, create_convex_driving_corridor, create_concave_driving_corridor, project_footprint_to_corridor, visualize_corridor
from findcontours import findObstaclePolygons, drawPolygons
from occupany_trajectory_polygon_check import check_footprints_to_poly_intersect, drawIntersection 

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def split_into_chunks(lst, n_chunks):
    chunk_size = len(lst) // n_chunks
    remainder = len(lst) % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

if __name__ == "__main__":
    # Physical dimensions of the grid map
    Lx = 25  # Length in meters
    Ly = 25   # Width in meters

    # Grid dimensions
    Nx = 100  # Number of cells along the length
    Ny = 100  # Number of cells along the width

    # Calculate voxel size
    vx, vy = calculate_voxel_size(Lx, Ly, Nx, Ny)

    # Create a 2D grid map
    grid_map = create_grid_map(Lx, Ly, Nx, Ny)

    # Simulate a simple trajectory
    points = np.array([(20.0, 1.0), (15.0, 5.0), (17.0, 12.0), (23.0, 20.0)])  # List of (x, y) waypoints
    points = interpolate_waypoints(points, num_points=3000)
    trajectory = posesFromPoints(points)

    robot_length = 1.0  # meters
    robot_width = 0.5   # meters
    robot_footprint = create_robot_footprint(robot_length, robot_width)
    ##############################################################################
    t0 = time.time()
    t = time.time()
    print(f"projection poly dt: {time.time() - t}")
    t = time.time()
    obstacles = findObstaclePolygons(grid_map, (vx,vy))
    print(f"obstacle poly dt: {time.time() - t}")
    t = time.time()
    n_cores = 4
    chunks = split_into_chunks(trajectory, n_cores)
    def work(data):
        trajectory, obstacles, robot_footprint = data
        projected_footprint_polygon = project_footprint_to_corridor(trajectory, robot_footprint, downsample=5)
        intersections = check_footprints_to_poly_intersect(projected_footprint_polygon, obstacles)
        return projected_footprint_polygon, intersections
    start = time.perf_counter()
    args = [(trajectory, obstacles, robot_footprint) for trajectory in chunks]
    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(work, args))
    footprints_all, intersections_all = zip(*results)
    footprints_combined = [item for item in footprints_all]
    intersections_combined = [item for sublist in intersections_all for item in sublist]
    end = time.perf_counter()
    print(f"footprints check dt: {end - start}")
    print(f"=======================\nsum dt: {time.time() - t0}")
    ##### plots
    fig, ax = plt.subplots(1, 1)
    ax = drawTrajectory(ax, points)
    for footprint in footprints_combined:
        ax = visualize_corridor(ax, footprint)
    ax = drawPolygons(ax, obstacles)
    for i in intersections_combined:
        ax = drawIntersection(ax, i)
    plt.show()

