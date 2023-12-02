import numpy as np
from itertools import permutations, product

def rotation_matrix_from_angles(x_deg, y_deg, z_deg):
    # Convert angles to radians
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)
    z_rad = np.radians(z_deg)

    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x_rad), -np.sin(x_rad)],
                   [0, np.sin(x_rad), np.cos(x_rad)]])

    Ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
                   [0, 1, 0],
                   [-np.sin(y_rad), 0, np.cos(y_rad)]])

    Rz = np.array([[np.cos(z_rad), -np.sin(z_rad), 0],
                   [np.sin(z_rad), np.cos(z_rad), 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx)).astype(np.int8)

    return rotation_matrix

def generate_rotation_matrices():
    # Values that can be placed in the matrix
    values = [0.0, 90.0, 180.0, 270.0]
    combinations = list(product(values, repeat=3))
    r = []
    for x,y,z in combinations:
        # Calculate rotation matrix
        r.append(rotation_matrix_from_angles(x, y, z))
    return r

if __name__ == '__main__':
    rot = generate_rotation_matrices()
    for r in rot: 
        print(r)

