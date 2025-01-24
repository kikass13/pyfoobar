import time 
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

from occupancy_trajectory_check import calculate_voxel_size, create_grid_map, interpolate_trajectory, posesFromPoints, drawTrajectory
from driving_corridor import create_robot_footprint, create_convex_driving_corridor, create_concave_driving_corridor, project_footprint_to_corridor, visualize_corridor
from findcontours import findObstaclePolygons, drawPolygons



def check_footprint_to_poly_intersect(p1, others):
    intersections = []
    for p2 in others:
        intersections.append(p1.intersection(p2))
    return intersections

def drawIntersection(ax, intersection):
    # Plot the intersection
    if intersection.is_valid:
        if intersection.geom_type == 'Polygon':
            x_int, y_int = intersection.exterior.xy
            ax.fill(x_int, y_int, color='purple', alpha=0.7, label='Intersection')
            ax.plot(x_int, y_int, color='purple', linewidth=2)
        elif intersection.geom_type == 'MultiPolygon':
            for poly in intersection.geoms:
                x_int, y_int = poly.exterior.xy
                ax.fill(x_int, y_int, color='purple', alpha=0.7, label='Intersection')
                ax.plot(x_int, y_int, color='purple', linewidth=2)
        elif intersection.geom_type == 'LineString':
            x_int, y_int = intersection.xy
            ax.plot(x_int, y_int, color='purple', linewidth=2, label='Intersection')
    return ax

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
    points = interpolate_trajectory(points, num_points=300)
    trajectory = posesFromPoints(points)

    robot_length = 1.0  # meters
    robot_width = 0.5   # meters
    robot_footprint = create_robot_footprint(robot_length, robot_width)
    ##############################################################################
    t0 = time.time()
    t = time.time()
    footprint_polygons = project_footprint_to_corridor(trajectory, robot_footprint, downsample=5)
    print(f"corridor dt: {time.time() - t}")
    t = time.time()
    obstacles = findObstaclePolygons(grid_map, (vx,vy))
    print(f"obstacle poly dt: {time.time() - t}")
    t = time.time()
    intersections = check_footprint_to_poly_intersect(footprint_polygons, obstacles)
    print(f"intersect check dt: {time.time() - t}")
    print(f"=======================\nsum dt: {time.time() - t0}")
    ##### plots
    fig, ax = plt.subplots(1, 1)
    ax = drawTrajectory(ax, points)
    ax = visualize_corridor(ax, footprint_polygons)
    ax = drawPolygons(ax, obstacles)
    for i in intersections:
        ax = drawIntersection(ax, i)
    plt.show()

