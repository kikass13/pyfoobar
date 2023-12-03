import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_colored_cube_array(N=20, size=1.0):
    def generate_side(N, side_center, side_orientation, side_color):
        # Calculate the rotation matrix using Rodrigues' rotation formula
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        dot_product = np.dot(z_axis, side_orientation) / (np.linalg.norm(z_axis) * np.linalg.norm(side_orientation))
        # Check if the side is parallel to the z-axis (for front and back)
        if np.isclose(np.abs(dot_product), 1):
            rotation_axis = np.array([0, 0, 1], dtype=np.float32)  # Use x-axis as the rotation axis
        else:
            rotation_axis = np.cross(z_axis, side_orientation)
        # Calculate the rotation matrix using Rodrigues' rotation formula
        rotation_axis /= np.linalg.norm(rotation_axis)
        theta = np.arccos(dot_product)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        # Generate 2D grid in local coordinates
        x = np.linspace(-0.5, 0.5, N)
        y = np.linspace(-0.5, 0.5, N)
        grid_x, grid_y = np.meshgrid(x, y)
        # Convert local coordinates to global coordinates using the rotation matrix
        local_coordinates = np.column_stack((grid_x.flatten(), grid_y.flatten(), np.zeros_like(grid_x.flatten())))
        global_coordinates = side_center + np.dot(local_coordinates, rotation_matrix.T)
        return global_coordinates, np.tile(side_color, (len(global_coordinates),1))
    # Generate colored sides
    side_front, color_front = generate_side(N, [0, 0, 0.5],   np.array([0, 0, 1]),  [1.0, 0.0, 0.0])  # Red
    side_back, color_back = generate_side(N, [0, 0, -0.5],    np.array([0, 0, -1]), [0.0, 1.0, 0.0])  # Green
    side_right, color_right = generate_side(N, [0.5, 0, 0],   np.array([1, 0, 0]),  [0.0, 0.0, 1.0])  # Blue
    side_left, color_left = generate_side(N, [-0.5, 0, 0],    np.array([-1, 0, 0]), [1.0, 1.0, 0.0])  # Yellow
    side_top, color_top = generate_side(N, [0, 0.5, 0],       np.array([0, 1, 0]),  [1.0, 0.0, 1.0])  # Magenta
    side_bottom, color_bottom = generate_side(N, [0, -0.5, 0],np.array([0, -1, 0]), [0.0, 1.0, 1.0])  # Cyan

    ### draw up line
    arrorx = np.tile(0.0, (250,1))
    arrory = np.tile(0.0, (250,1))
    arrowz = np.linspace(0, 1.0, 250)
    arrow = np.column_stack((arrorx, arrory, arrowz))
    arrowcolor = np.tile(np.array([0.7, 0.7, 0.7]), (250,1))

    # Combine all sides
    all_side_points = np.concatenate([side_front, side_back, side_right, side_left, side_top, side_bottom, arrow], dtype=np.float32)
    all_side_colors = np.concatenate([color_front, color_back, color_right, color_left, color_top, color_bottom, arrowcolor], dtype=np.float32)
    # all_side_points = np.concatenate([side_front ])
    # all_side_colors = np.concatenate([color_front])

    # Scale and return the colored cube sides
    return all_side_points * size, all_side_colors

# Function to plot all points in the colored cube
def plot_colored_cube_array(N=20, size=1.0):
    cube_side_points, cube_side_colors = create_colored_cube_array(N=N, size=size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for all points
    ax.scatter(cube_side_points[:, 0], cube_side_points[:, 1], cube_side_points[:, 2], c=cube_side_colors, marker='o')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-size * 2.0, size * 2.0)
    ax.set_ylim(-size * 2.0, size * 2.0)
    ax.set_zlim(-size * 2.0, size * 2.0)

    plt.show()

if __name__ == '__main__':
    # Example usage
    plot_colored_cube_array(N=20, size=2.0)