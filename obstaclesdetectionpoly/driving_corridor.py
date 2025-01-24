

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import matplotlib.pyplot as plt

from occupancy_trajectory_check import drawTrajectory

def create_robot_footprint(length, width):
    """
    Create a rectangular footprint centered at (0, 0).
    :param length: Length of the robot (front to back).
    :param width: Width of the robot (side to side).
    :return: Shapely Polygon representing the footprint.
    """
    half_length = length / 2
    half_width = width / 2
    return Polygon([
        (-half_length, -half_width), (half_length, -half_width),
        (half_length, half_width), (-half_length, half_width),
        (-half_length, -half_width)
    ])

def transform_footprint(footprint, pose):
    """
    Transform the robot footprint based on its pose.
    :param footprint: Shapely Polygon of the robot's footprint.
    :param pose: Tuple (x, y, theta) where x, y are position, and theta is orientation (in radians).
    :return: Transformed Shapely Polygon.
    """
    x, y, theta = pose
    rotated = rotate(footprint, np.degrees(theta), origin=(0, 0), use_radians=True)
    translated = translate(rotated, xoff=x, yoff=y)
    return translated

def create_concave_driving_corridor(trajectory, footprint):
    """
    Project the robot footprint along the trajectory and create a driving corridor.
    :param trajectory: List of poses [(x, y, theta), ...].
    :param footprint: Shapely Polygon of the robot's footprint.
    :return: Shapely Polygon representing the driving corridor.
    """
    import alphashape
    footprints = [transform_footprint(footprint, pose) for pose in trajectory]
    combined_polygons = unary_union(footprints)  # Combine into a single polygon
    if isinstance(combined_polygons, MultiPolygon):
        points = []
        for poly in combined_polygons.geoms:
            x, y = poly.exterior.xy
            points.extend(zip(x, y))
    elif isinstance(combined_polygons, Polygon):
        x, y = combined_polygons.exterior.xy
        points = list(zip(x, y))
    concave_hull = alphashape.alphashape(points, 0.0)
    return combined_polygons, concave_hull
def create_convex_driving_corridor(trajectory, footprint):
    """
    Project the robot footprint along the trajectory and create a driving corridor.
    :param trajectory: List of poses [(x, y, theta), ...].
    :param footprint: Shapely Polygon of the robot's footprint.
    :return: Shapely Polygon representing the driving corridor.
    """
    combined_polygons = project_footprint_to_corridor(trajectory, footprint)
    convex_hull = combined_polygons.convex_hull
    return combined_polygons, convex_hull

def project_footprint_to_corridor(trajectory, footprint, downsample=None):
    if downsample:
        trajectory = trajectory[::downsample]
    footprints = [transform_footprint(footprint, pose) for pose in trajectory]
    combined_polygons = unary_union(footprints)  # Combine into a single polygon
    inflated = combined_polygons
    ### inflation takes a loooong time ... damn
    # if inflation:
    #     inflated = combined_polygons.buffer(inflation, join_style=2)
    return inflated

def visualize_corridor(ax, footprint_polygons, corridor=None):
    """
    Visualize the driving corridor and the trajectory.
    :param corridor: Shapely Polygon of the driving corridor.
    :param trajectory: List of poses [(x, y, theta), ...].
    """

    if corridor:
        x, y = corridor.exterior.xy
        ax.fill(x, y, color='lightblue', alpha=0.5, label='Driving Corridor')

    # Plot each polygon in the MultiPolygon
    if isinstance(footprint_polygons, MultiPolygon):
        for polygon in footprint_polygons.geoms:  # Access individual polygons
            x, y = polygon.exterior.xy  # Get the exterior boundary
            ax.fill(x, y, color='lightblue', alpha=0.5, label='Driving Corridor' if polygon == footprint_polygons.geoms[0] else None)
            ax.plot(x, y, color='blue', linewidth=1)
    else:
        x, y = footprint_polygons.exterior.xy  # Get the exterior coordinates of the single polygon
        ax.fill(x, y, color='lightblue', alpha=0.5, label='Combined Polygon')
        ax.plot(x, y, color='blue', linewidth=1)  # Draw the boundary
    return ax

if __name__ == "__main__":
    # Robot parameters
    robot_length = 1.0  # meters
    robot_width = 0.5   # meters

    # Trajectory: [(x, y, theta), ...]
    trajectory = np.array([
        (0, 0, 0),
        (1, 0.5, np.pi / 8),
        (2, 1, np.pi / 6),
        (3, 1.5, np.pi / 4)
    ])

    # Create robot footprint
    robot_footprint = create_robot_footprint(robot_length, robot_width)

    # Generate the driving corridor
    footprint_polygons, hull = create_convex_driving_corridor(trajectory, robot_footprint)

    fig, ax = plt.subplots(1, 1)
    # Visualize the result
    ax = visualize_corridor(ax, footprint_polygons, hull)
    ax = drawTrajectory(ax, trajectory[:,:2])

    plt.axis('equal')
    plt.legend()
    plt.title("Driving Corridor Along Trajectory")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()

