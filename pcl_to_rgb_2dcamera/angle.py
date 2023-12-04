import numpy as np
from scipy.spatial.transform import Rotation as Rot

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle
    Parameters
    ----------
    angle :
    Returns
    -------
    A 2D rotation matrix
    Examples
    --------
    >>> angle_mod(-4.0)
    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)
    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.
    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.
    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531
    >>> angle_mod([-4.0])
    np.array(2.28318531)
    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])
    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])
    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle
    
def rodrigues(rotation_vector):
    # Normalize the rotation vector
    norm = np.linalg.norm(rotation_vector)
    if norm == 0:
        return np.eye(3)
    rotation_axis = rotation_vector / norm
    theta = norm
    # Rodrigues' rotation formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cross_matrix = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + sin_theta * cross_matrix + (1 - cos_theta) * np.dot(cross_matrix, cross_matrix)
    return rotation_matrix

def euler_to_forward_vector(euler_angls, axis=0):
    # Convert Euler angles to rotation matrix
    rol,pitch,yaw = euler_angls
    rotation_matrix = rodrigues(np.array([rol, pitch, yaw]))
    # Extract forward vector (third column of the rotation matrix)
    forward_vector = rotation_matrix[:, axis]
    return forward_vector


def ForwardUpDown_rotation_matrix_from_angles(x_rad, y_rad, z_rad):
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

def generate_all_projection_rotation_matrices():
    from itertools import product
    # Values that can be placed in the matrix
    # deg_values = [-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0]
    deg_values = [0.0, 90.0, 180.0, 270.0]
    rad_values = np.radians(deg_values)
    combinations = list(product(rad_values, repeat=3))
    r = []
    for rol,pit,yaw in combinations:
        # Calculate rotation matrix
        euler = (rol, pit, yaw)
        rotation = ForwardUpDown_rotation_matrix_from_angles(rol,pit,yaw)
        r.append((rotation, euler))
    return r

if __name__ == '__main__':
    import numpy as np
    import cv2
    roll_angle = np.radians(0)  # Example roll angle in radians
    pitch_angle = np.radians(0)  # Example yaw angle in radians
    yaw_angle = np.radians(0)  # Example pitch angle in radians
    forward_vector1 = euler_to_forward_vector([roll_angle, pitch_angle, yaw_angle])
    print("1 Forward Vector:", forward_vector1)

    roll_angle = np.radians(0)  # Example roll angle in radians
    pitch_angle = np.radians(0)  # Example yaw angle in radians
    yaw_angle = np.radians(90)  # Example pitch angle in radians
    forward_vector2 = euler_to_forward_vector([roll_angle, pitch_angle, yaw_angle])
    print("2 Forward Vector:", forward_vector2)

    allPossibilities = generate_all_projection_rotation_matrices()
    for i, (rotation_matrix, euler) in enumerate(allPossibilities):
        print("===========")
        print(i)
        print(np.degrees(euler))
        print(rotation_matrix)
    print("==================================================================")
    print("==================================================================")
    print("==================================================================")
    print("specific")
    for mat, euler in allPossibilities:
        if mat[0,1] == -1 and mat[1,2] == 1 and mat[2,0] == -1:
            print(np.degrees(euler))
    print("==================================================================")
    print("==================================================================")
    print("==================================================================")
    sampleRot = generate_all_projection_rotation_matrices()[22][0]
    roll = np.radians(0)  # Example roll angle in radians
    pitch = np.radians(0)  # Example yaw angle in radians
    yaw = np.radians(90)  # Example pitch angle in radians
    newRot = rodrigues(np.array([roll, pitch, yaw]))
    print(sampleRot)
    print(" ****** ")
    print(np.round(newRot))
    result = -np.dot(sampleRot, newRot)
    print(" = ")
    print(np.round(result))