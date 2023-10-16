import open3d as o3d
import numpy as np


def find_circle_in_cuboid(W, D, H):
    # Calculate the center of the cuboid
    Center_X = W / 2
    Center_Y = D / 2
    Center_Z = H / 2

    # Calculate the radius of the circle
    # Radius = min(Center_X, Center_Y, Center_Z)
    Radius = max(Center_X, Center_Y, Center_Z)

    # Calculate the center of the circle
    Circle_Center = (Center_X, Center_Y, Center_Z)

    return Circle_Center, Radius

def projectCircleFrom3dPoints(points):
    # Define a plane (for projection)
    plane_origin = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])

    # Project 3D points onto the plane
    projected_points = points - np.outer(np.dot(points - plane_origin, plane_normal), plane_normal)

    # Create a 2D circle
    center = np.mean(projected_points, axis=0)
    radius = np.max(np.linalg.norm(projected_points - center, axis=1))

    # Visualize the 2D circle
    circle = o3d.geometry.TriangleMesh.create_cylinder(radius, 0.00001, 100)
    circle.compute_vertex_normals()
    circle.translate(center)
    return circle

def projectCircleFromCuboid(center, w,d,h, color):
    # Create a 2D circle
    corrected_center , radius = find_circle_in_cuboid(w,d,h)
    center = center + corrected_center
    # Visualize the 2D circle
    circle = o3d.geometry.TriangleMesh.create_cylinder(radius, 0.00001, 100)
    circle.paint_uniform_color(color)
    circle.compute_vertex_normals()
    circle.translate(center)
    return circle



if __name__ == '__main__':
    # Create some 3D points
    points = np.array([[1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0]])

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    circle = projectCircleFrom3dPoints(points)

    # Add the circle to the visualization
    vis.add_geometry(circle)

    # Set the view control to see the circle
    vis.get_view_control().set_constant_z_far(100)

    # Run the visualization
    vis.run()

    # Close the visualization window
    vis.destroy_window()