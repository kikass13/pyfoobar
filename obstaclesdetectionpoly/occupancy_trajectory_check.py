import time 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Function to calculate voxel size
def calculate_voxel_size(Lx, Ly, Nx, Ny):
    vx = Lx / Nx  # Voxel size along X-axis
    vy = Ly / Ny  # Voxel size along Y-axis
    return vx, vy

# Function to create a 2D grid map with a natural environment-like layout
def create_grid_map(Lx, Ly, Nx, Ny):
    # Create a 2D grid map with specific obstacles and road-like structures
    grid_map = np.zeros((Nx, Ny), dtype=int)  # 0 for free, 1 for occupied
    # Buildings (blocks)
    for i in range(Nx // 3, 2 * Nx // 3):
        for j in range(3 * Ny // 4, Ny):
            grid_map[i, j] = 1
    for i in range(Nx // 6, 2 * Nx // 6):
        for j in range(3 * Ny // 8, Ny):
            grid_map[i, j] = 1
    for i in range(0, 20):
        for j in range(0, 20):
            grid_map[i, j] = 1
    return grid_map.astype(np.uint8)

# Function to check if a point is within the grid
def is_within_grid(x, y, Nx, Ny):
    return 0 <= x < Nx and 0 <= y < Ny

def point_to_grid_index(point, voxel_size, grid_origin=(0, 0)):
    x, y = point[:2]
    # Convert point to grid index
    grid_x = int((x - grid_origin[0]) / voxel_size[0])
    grid_y = int((y - grid_origin[1]) / voxel_size[1])
    # if len(point) == 3:  # 3D case
    #     z = point[2]
    #     grid_z = int((z - grid_origin[2]) / voxel_size[2])
    #     return grid_z, grid_x, grid_y
    # else:  # 2D case
    return grid_x, grid_y

# Function to check if a path intersects with occupied voxels in the grid
def check_path(grid_map, path, voxel_size):
    status = []  # List to store the status for each point
    for p in path:
        x,y = p
        ### invert x,y cause reasons
        grid_y, grid_x = point_to_grid_index(p, voxel_size)
        if not is_within_grid(grid_x, grid_y, grid_map.shape[0], grid_map.shape[1]):
            status.append(True)  # Out of grid boundaries
        elif grid_map[grid_x, grid_y] == 1:
            status.append(False)  # Intersects with an occupied voxel
        else:
            status.append(True)  # Free space
    return status

def interpolate_waypoints(waypoints, num_points=10):
    x, y = zip(*waypoints)  # Separate into x and y
    # Interpolating to n points
    x_new = np.linspace(min(x), max(x), num_points)  # Generate n evenly spaced x-values
    y_interpolator = interp1d(x, y, kind='linear')  # Linear interpolation for y
    y_new = y_interpolator(x_new)
    interpolated_points = list(zip(x_new, y_new))
    return interpolated_points

def posesFromPoints(points, invertxy=False):
    poses = []
    for i in range(len(points)-1):
        x, y = points[i]
        x_next, y_next = points[i + 1]
        ### invert x and y cause of image coordinates?
        if invertxy:
            theta = np.arctan2(x_next - x, y_next - y)
        else:
            theta = np.arctan2(y_next - y, x_next - x)
        poses.append((x,y,theta))
    poses.append((points[-1][0], points[-1][1], theta))
    return np.vstack([poses])

def drawGridmap(ax, grid_map, grid_size):
    Lx, Ly = grid_size
    ax.imshow(grid_map, cmap='gray', origin='lower', extent=(0, Lx, 0, Ly))
    return ax
def drawTrajectory(ax, trajectory, path_status=None):
    # Plot the trajectory and mark each point with its status
    for idx, (x, y) in enumerate(trajectory):
        color = 'green' if path_status and path_status[idx] else 'red'  # Free (green) or blocked (red)
        ax.plot(x, y, '.', color=color)
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
    trajectory = np.array([(20.0, 1.0), (15.0, -2.0), (17.0, 12.0), (23.0, 20.0)])  # List of (x, y) waypoints
    trajectory = interpolate_waypoints(trajectory, num_points=1000)
    # Check if the adjusted trajectory intersects with any occupied voxels and get the status for each point
    t = time.time()
    path_status = check_path(grid_map, trajectory, (vx,vy))
    print(f"{time.time() - t}")

    ##############################################################################

    # Visualization of the grid map
    # fig = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(1, 1)
    ax = drawGridmap(ax, grid_map, (Lx, Ly))
    ax = drawTrajectory(ax, trajectory)
    plt.title(f'2D Grid Map with Voxel Size (vx, vy) = ({vx:.2f} m, {vy:.2f} m)')
    plt.show()
