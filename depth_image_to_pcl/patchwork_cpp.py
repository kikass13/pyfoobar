import os
import sys
import open3d as o3d
import numpy as np
# import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
input_cloud_filepath = os.path.join(cur_dir, "cloud.pcd")

try:
    home = os.path.expanduser('~')
    patchwork = os.path.join(home, "gitclones", "patchwork-plusplus")
    patchwork_module_path = os.path.join(patchwork, "build/python_wrapper")
    print(patchwork_module_path)
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)


if __name__ == "__main__":
    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
#   Params() {
#         verbose     = false;
#         enable_RNR  = true;
#         enable_RVPF = true;
#         enable_TGR  = true;

#         num_iter = 3;               // Number of iterations for ground plane estimation using PCA.
#         num_lpr = 20;               // Maximum number of points to be selected as lowest points representative.
#         num_min_pts = 10;           // Minimum number of points to be estimated as ground plane in each patch.
#         num_zones = 4;              // Setting of Concentric Zone Model(CZM)
#         num_rings_of_interest = 4;  // Number of rings to be checked with elevation and flatness values.

#         RNR_ver_angle_thr = -15.0;  // Noise points vertical angle threshold. Downward rays of LiDAR are more likely to generate severe noise points.
#         RNR_intensity_thr = 0.2;    // Noise points intensity threshold. The reflected points have relatively small intensity than others.
        
#         sensor_height = 1.723;      
#         th_seeds = 0.125;           // threshold for lowest point representatives using in initial seeds selection of ground points.
#         th_dist = 0.125;            // threshold for thickenss of ground.
#         th_seeds_v = 0.25;          // threshold for lowest point representatives using in initial seeds selection of vertical structural points.
#         th_dist_v = 0.1;            // threshold for thickenss of vertical structure.
#         max_range = 80.0;           // max_range of ground estimation area
#         min_range = 2.7;            // min_range of ground estimation area
#         uprightness_thr = 0.707;    // threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
#         adaptive_seed_selection_margin = -1.2; // parameter using in initial seeds selection

#         num_sectors_each_zone = {16, 32, 54, 32};   // Setting of Concentric Zone Model(CZM)
#         num_rings_each_zone = {2, 4, 4, 4};         // Setting of Concentric Zone Model(CZM)

#         max_flatness_storage = 1000;    // The maximum number of flatness storage
#         max_elevation_storage = 1000;   // The maximum number of elevation storage
#         elevation_thr = {0, 0, 0, 0};   // threshold of elevation for each ring using in GLE. Those values are updated adaptively.
#         flatness_thr = {0, 0, 0, 0};    // threshold of flatness for each ring using in GLE. Those values are updated adaptively.
#     }
# };

    params.verbose = True
    params.min_range = 0.1
    params.max_range = 80.0
    params.sensor_height = -0.5
    params.uprightness_thr = 0.707
    params.th_dist = 0.5
    params.elevation_thr:  [0.5, 0.8, 1.0, 1.1]  # For flatness
    params.flatness_thresholds:  [0.0, 0.000125, 0.000185, 0.000185]  # For flatness
    params.enable_RVPF = False
    # params.enable_TGR = False

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)


    ### random depth image to pcl
    pointcloud = o3d.io.read_point_cloud("cloud0.pcd")
    pointcloud = np.asarray(pointcloud.points)
    xaxa = pointcloud.T
    # xaxa[[0, 1, 2]] = xaxa[[0, 1, 2]]   ###swap axis dimensions of points
    # xaxa[[0, 1, 2]] = xaxa[[0, 2, 1]]   ###swap axis dimensions of points
    # xaxa[[0, 1, 2]] = xaxa[[1, 0, 2]]   ###swap axis dimensions of points
    # xaxa[[0, 1, 2]] = xaxa[[1, 2, 0]]   ###swap axis dimensions of points
    # xaxa[[0, 1, 2]] = xaxa[[2, 1, 0]] ###swap axis dimensions of points
    xaxa[[0, 1, 2]] = xaxa[[2, 0, 1]] ###swap axis dimensions of points
    pointcloud = xaxa.T

    ###invert x & z axis
    # print(pointcloud)
    # print("===================================")
    pointcloud[:,2] *= -1.0
    # print(pointcloud)
    ### filter z (not really needed anymore)
    filtered = []
    for x,y,z in pointcloud:
        if z < 2.0 and x > 15:     ### dont look up, i guess xD
            filtered.append([x,y,z])
    pointcloud = np.array(filtered)

    # ### eth pcl
    # pointcloud = o3d.io.read_point_cloud("eth3d_cloud0.pcd")
    # pointcloud = np.asarray(pointcloud.points)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(pointcloud)
    PatchworkPLUSPLUS.estimateGround(pointcloud)
    PatchworkPLUSPLUS.estimateGround(pointcloud)
    PatchworkPLUSPLUS.estimateGround(pointcloud)
    PatchworkPLUSPLUS.estimateGround(pointcloud)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    nonground   = PatchworkPLUSPLUS.getNonground()
    time_taken  = PatchworkPLUSPLUS.getTimeTaken()

    # Get centers and normals for patches
    centers     = PatchworkPLUSPLUS.getCenters()
    normals     = PatchworkPLUSPLUS.getNormals()

    print("Origianl Points  #: ", pointcloud.shape[0])
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

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(centers_o3d)

    print(o3d.io.write_point_cloud("nonground.pcd", nonground_o3d, write_ascii=True, compressed=False, print_progress=True))


    downpcd = nonground_o3d.voxel_down_sample(voxel_size=0.5)
    print(o3d.io.write_point_cloud("nonground_downsampled.pcd", downpcd, write_ascii=True, compressed=False, print_progress=True))



    vis.run()
    vis.destroy_window()
