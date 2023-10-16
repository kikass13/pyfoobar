import os
import sys
import open3d as o3d
import numpy as np
# import pypatchworkpp

import cluster
import circle_from_3d_ponts

cur_dir = os.path.dirname(os.path.abspath(__file__))
home = os.path.expanduser('~')

try:
    patchwork = os.path.join(home, "gitclones", "patchwork-plusplus")
    patchwork_module_path = os.path.join(patchwork, "build/python_wrapper")
    print(patchwork_module_path)
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)


if __name__ == "__main__":
    params = pypatchworkpp.Parameters()

    params.verbose = True
    params.min_range = 0.1
    params.max_range = 80.0
    params.sensor_height = -0.5
    params.uprightness_thr = 0.707
    params.th_dist = 0.25
    params.elevation_thr:  [0.5, 0.8, 1.0, 1.1]  # For flatness
    params.flatness_thresholds:  [0.0, 0.000125, 0.000185, 0.000185]  # For flatness
    params.enable_RVPF = False
    # params.enable_TGR = False

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)



    def read_bin(bin_path):
        scan = np.fromfile(bin_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan


    #### read some files:
    files = [ 
        "000001.bin",
        "000002.bin",
        "000003.bin",
        # "000004.bin",
        # "000005.bin",
        # "000006.bin",
        # "000007.bin",
        # "000008.bin",
        # "000009.bin",
        # "000010.bin",
        # "000011.bin",
        # "000012.bin",
        # "000013.bin",
        # "000014.bin",
        # "000015.bin",
        # "000016.bin",
        # "000017.bin",
        # "000018.bin",
        # "000019.bin",
        # "000020.bin",
    ]
    clouds = []
    for fpath in files:
        path = os.path.join(home, "Downloads", "data_object_velodyne", "testing", "velodyne", fpath)
        pcl = read_bin(path)
        clouds.append(pcl)

    ########################################################

    for pcl in clouds:
        # Estimate Ground
        PatchworkPLUSPLUS.estimateGround(pcl)

        # Get Ground and Nonground
        ground      = PatchworkPLUSPLUS.getGround()
        nonground   = PatchworkPLUSPLUS.getNonground()
        time_taken  = PatchworkPLUSPLUS.getTimeTaken()

        # Get centers and normals for patches
        centers     = PatchworkPLUSPLUS.getCenters()
        normals     = PatchworkPLUSPLUS.getNormals()

        # print("Origianl Points  #: ", pointcloud.shape[0])
        print("Ground Points    #: ", ground.shape[0])
        print("Nonground Points #: ", nonground.shape[0])
        print("Time Taken : ", time_taken / 1000000, "(sec)")
        print("Press ... \n")
        print("\t H  : help")
        print("\t N  : visualize the surface normals")
        print("\tESC : close the Open3D window")

        # Visualize
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width = 1024, height = 768)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        ground_o3d = o3d.geometry.PointCloud()
        if ground.size != 0:
            ground_o3d.points = o3d.utility.Vector3dVector(ground)
            ground_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
            )

        nonground_o3d = o3d.geometry.PointCloud()
        if nonground.size != 0:
            nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
            nonground_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
            )

        centers_o3d = o3d.geometry.PointCloud()
        if centers.size != 0:
            centers_o3d.points = o3d.utility.Vector3dVector(centers)
            centers_o3d.normals = o3d.utility.Vector3dVector(normals)
            centers_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
            )

        # print(o3d.io.write_point_cloud("nonground.pcd", nonground_o3d, write_ascii=True, compressed=False, print_progress=True))
        downpcd = nonground_o3d.voxel_down_sample(voxel_size=0.5)
        # print(o3d.io.write_point_cloud("nonground_downsampled.pcd", downpcd, write_ascii=True, compressed=False, print_progress=True))

        objects, colored, unclustered = cluster.cluster(downpcd, 0.05, min_samples=3, only_xy_dist=True)
        cuboids = cluster.createCuboids(objects)
        smallObjects = []
        for o in objects:
            center,w,d,h,col = o
            volume = w*d*h 
            if h > 0.5 and h < 3.0 and volume > 0.5 and volume < 3.0:
                smallObjects.append(o)

        for cub in cuboids:
            vis.add_geometry(cub)
        for center,w,d,h,col in smallObjects:
            circle = circle_from_3d_ponts.projectCircleFromCuboid(center, w,d,h, col)
            vis.add_geometry(circle)


        # vis.add_geometry(mesh)
        # vis.add_geometry(ground_o3d)
        # vis.add_geometry(nonground_o3d)
        # vis.add_geometry(centers_o3d)
        vis.add_geometry(colored)

        vis.run()
        vis.destroy_window()