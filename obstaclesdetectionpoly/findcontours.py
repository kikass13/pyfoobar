

import numpy as np
import cv2
import time
import shapely
import matplotlib.pyplot as plt

from occupancy_trajectory_check import calculate_voxel_size, create_grid_map, drawGridmap


def findObstaclePolygons(grid_map, voxel_size):
    vx, vy = voxel_size
    contours, _ = cv2.findContours(grid_map, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Convert pixel coordinates to world coordinates
        world_coords = [(pt[0][0] * vx, pt[0][1] * vy) for pt in contour]
        polygons.append(shapely.Polygon(world_coords))
    return polygons
def drawPolygons(ax, polygons):
    for p in polygons:
        x, y = p.exterior.xy
        ax.plot(x, y, color='red', linewidth=2, label='Polygon')
        ax.fill(x, y, color='pink', alpha=0.5)  # Fill the polygon
        ax.scatter(x, y, color='red')  # Mark the vertices
    return ax

if __name__ == "__main__":
    Lx = 25  # Length in meters
    Ly = 25   # Width in meters

    # Grid dimensions
    Nx = 100  # Number of cells along the length
    Ny = 100  # Number of cells along the width

    # Calculate voxel size
    vx, vy = calculate_voxel_size(Lx, Ly, Nx, Ny)

    # Create a 2D grid map
    grid_map = create_grid_map(Lx, Ly, Nx, Ny).astype(np.uint8)

    t = time.time()
    polygons = findObstaclePolygons(grid_map, (vx,vy))
    print(f"polygon finder dt: {time.time() - t}")


    fig, ax = plt.subplots(1, 1)
    ax = drawGridmap(ax, grid_map, (Lx, Ly))
    ax = drawPolygons(ax, polygons)
    plt.title("Shapely Polygon")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.axis('equal')  # Equal scaling for X and Y axes
    plt.show()