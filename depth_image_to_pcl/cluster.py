import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import time

from sympy import symbols, Eq, solve


### pip3 install pyransac3d
import pyransac3d as pyrsc

def cluster(pcd, epsilon, min_samples=10, algorithm='ball_tree', n_jobs=2, only_xy_dist=False):
    start = time.time()

    # Get points and transform it to a numpy array:
    points = np.asarray(pcd.points)

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points)

    # Clustering:
    ### general:
    ### https://github.com/christianversloot/machine-learning-articles/blob/main/performing-dbscan-clustering-with-python-and-scikit-learn.md
    ### ball tree vs kd tree:
    ### https://scikit-learn.org/stable/modules/neighbors.html
    # model = DBSCAN(eps=0.3, min_samples=10, algorithm='kd_tree', n_jobs=2)
    if only_xy_dist:
        ### remove z axis, so that we dont cluster in z
        # Extract the xy-coordinates (first two columns) from both point sets
        scaled_points = scaled_points[:, :2]
        # distMat = distance_matrix(scaled_points, scaled_points, p=2)
        # model = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs, metric="precomputed")
        model = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs)
        # model.fit(distMat)
        model.fit(scaled_points)
    else:
        model = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs)
        model.fit(scaled_points)
    endcluster = time.time() - start

    start = time.time()
    # Get labels:
    labels = model.labels_
    # Get the number of colors:
    n_clusters = len(set(labels))
    # Mapping the labels classes to a color map:
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # Attribute to noise the black color:
    colors[labels < 0] = 0
    # Update points colors:
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    ### get all points of every cluster
    ### Create a dictionary to store points for each cluster
    # print("POINTS: %s" % len(points))
    clusters = {}
    for label in set(labels):
        cluster_points = points[labels == label]
        clusters[label] = cluster_points
    endextract = time.time() - start

    start = time.time()
    objects = []
    unclustered = []
    ### Now, 'clusters' contains the points belonging to each cluster, and you can access them as follows:
    for label, points in clusters.items():
        ### skip unclustered 
        if label == -1:
            unclustered = points
            continue

        # print(f"Cluster {label}:")
        # print(len(points))
        # print(points)

        ### if we have more than 5 points, the fitting can work
        ### if we have less, we just "assume" that all points are inside the "cuboid" and skip ransac
        if len(points) > 5:
            ### ransac sphere fitting
            # sph = pyrsc.Sphere()
            # center, radius, inliers = sph.fit(points, thresh=0.1, maxIteration=10)
            # print(center, radius)
            # objects.append((center, radius, inliers))
            cub = pyrsc.Cuboid()
            plane_equations, inliers = cub.fit(points, thresh=1.0, maxIteration=3)
            # print("in %s" % len(inliers))
            # print(plane_equations)

            # Create symbolic variables for the intersection point (x, y, z)
            # x, y, z = symbols('x y z')

            # # Define equations for the planes
            # eq1 = Eq(plane_equations[0, 0]*x + plane_equations[0, 1]*y + plane_equations[0, 2]*z + plane_equations[0, 3], 0)
            # eq2 = Eq(plane_equations[1, 0]*x + plane_equations[1, 1]*y + plane_equations[1, 2]*z + plane_equations[1, 3], 0)
            # eq3 = Eq(plane_equations[2, 0]*x + plane_equations[2, 1]*y + plane_equations[2, 2]*z + plane_equations[2, 3], 0)

            # ### Solve the system of equations to find the intersection point
            # intersection = solve((eq1, eq2, eq3), (x, y, z))

            # # Extract the coordinates of the intersection point
            # intersection_point = np.array([intersection[x], intersection[y], intersection[z]])
            # # print(intersection_point)

            # ### plane normals
            # plane_normals = plane_equations[:, :3]
            # print(plane_normals)
            # ### Calculate the dimensions of the cuboid
            # ### The dimensions are the lengths of the normal vectors
            # cuboid_width = np.linalg.norm(plane_normals[0])
            # cuboid_height = np.linalg.norm(plane_normals[1])
            # cuboid_depth = np.linalg.norm(plane_normals[2])

            # print(cuboid_width)
            # print(cuboid_height)
            # print(cuboid_depth)

            ### inliers are indices only, we want all pints of the given indices
            inliner_points = points[inliers]
        else:
            inliner_points = points

        ### find bounding box
        min_coords = np.min(inliner_points, axis=0)
        max_coords = np.max(inliner_points, axis=0)
        # center_coords = (min_coords + max_coords) 
        dimensions = max_coords - min_coords

        ### grab color
        color = plt.get_cmap("tab20")(label / (n_clusters if n_clusters > 0 else 1))[:3] #no alpha
        ### grab color of fist point in inliers list
        # color = colors[inliers[0]][:3] ###no alpha
        objects.append((min_coords, dimensions[0], dimensions[1], dimensions[2], color))
    endransac = time.time() - start

    ### visualize objects
    # for center, radius, inliers in objects:
    #     sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    #     sphere_mesh.translate(center)  # Set the position
    #     sphere_mesh.paint_uniform_color([0.8, 0.2, 0.2])  # Set the color
    #     vis.add_geometry(sphere_mesh)


    print(endcluster)
    print("+")
    print(endextract)
    print("+")
    print(endransac)
    print("=")
    print(endcluster + endextract + endransac)

    return objects, pcd, unclustered

def createCuboids(objects):
    cuboids = []
    for start, w,d,h, color in objects:
        #### width, height = x,y in open3d --- depth = z ???
        cuboid_mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=d, depth=h)
        cuboid_mesh.translate(start)
        # cuboid_mesh.paint_uniform_color(color[:3])
        # vis.add_geometry(cuboid_mesh)
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(cuboid_mesh)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(line_set.lines))
        cuboids.append(line_set)
    return cuboids

if __name__ == '__main__':
    # Read point cloud:
    # pcd = o3d.io.read_point_cloud("nonground.pcd")
    pcd = o3d.io.read_point_cloud("nonground_downsampled.pcd")


    objects, coloredPcd, unclustered = cluster(pcd, 0.3, only_xy_dist=False)
    cuboids = createCuboids(objects)

    volumes = [w*d*h for center,w,d,h,c in objects]
    
    # Display:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 1024, height = 768)
    for m in cuboids:
        vis.add_geometry(m)
    vis.add_geometry(coloredPcd)
    vis.run()
    vis.destroy_window()
