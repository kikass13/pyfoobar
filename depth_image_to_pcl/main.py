import open3d
import matplotlib.pyplot as plt
import numpy as np


def DisplayImage(input, colormap):
    plt.imshow(input, cmap=colormap)
    plt.show()

# Set camera intrinsic parmeters.
# intrinsic = open3d.camera.PinholeCameraIntrinsic()
# width (int) Width of the image.
# height (int) Height of the image.
# fx (float) X-axis focal length
# fy (float) Y-axis focal length.
# cx (float) X-axis principle point.
# cy (float) Y-axis principle point.
# intrinsic.set_intrinsics(850, 638, 790, 790, 0,0)
###default for sample data
# intrinsic = open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
# intrinsic = open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)

defaultExtrinsics = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
], dtype=np.float64)
### zed camera intrinsics
fx=700.819
fy=700.819
cx=665.465
cy=371.953
k1=-0.174318
k2=0.0261121
zedIntrinsic = open3d.camera.PinholeCameraIntrinsic()
zedIntrinsic.set_intrinsics(1000,1000, fx, fy, cx, cy)
### zed camera extrinsics
# Baseline=120
CV_2K=0.00958521
CV_FHD=0.00958521
CV_HD=0.00958521
CV_VGA=0.00958521
RX_2K=0.00497864
RX_FHD=0.00497864
RX_HD=0.00497864
RX_VGA=0.00497864
RZ_2K=-0.00185401
RZ_FHD=-0.00185401
RZ_HD=-0.00185401
RZ_VGA=-0.00185401
zedExtrinsics = np.array([ 
    [CV_2K, CV_FHD, CV_HD, CV_VGA],
    [RX_2K, RX_FHD, RX_HD, RX_VGA],
    [RZ_2K, RZ_FHD, RZ_HD, RZ_VGA],
    [0, 0, 0, 1]
], dtype=float)

# images = [open3d.data.SampleSUNRGBDImage().depth_path] #, open3d.data.SampleNYURGBDImage()
images = [
    # ("data/test/HR/outleft/0011.png", "data/test/HR/depthmap/0011.png", (zedIntrinsic, defaultExtrinsics)),
    (open3d.data.SampleSUNRGBDImage().color_path, open3d.data.SampleSUNRGBDImage().depth_path, (open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), defaultExtrinsics) ),
    # ("data/livingroom.jpg", "data/livingroom.jpg", (zedIntrinsic, defaultExtrinsics))
]

# Visualize
vis = open3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 600, height = 400)

for i, images in enumerate(images):
    # rgbd_image = open3d.io.read_image(image)
    color_raw_path, depth_raw_path, calibration = images
    intrinsics, extrinsics = calibration
    color_raw = open3d.io.read_image(color_raw_path)
    depth_raw = open3d.io.read_image(depth_raw_path)
    c = 1
    try:
        h,w,c = np.asarray(depth_raw).shape
    except:
        pass
    if c == 3:
        # depth_raw = np.dot(np.asarray(depth_raw)[...,:3], [0.333, 0.333, 0.333]) #### rgb 8 bit
        depth_raw = np.dot(np.asarray(depth_raw)[...,:3], [85, 85, 85]) ### rgb, 16 bit
        depth_raw = open3d.geometry.Image(depth_raw.astype(np.uint16))
    print(np.asarray(depth_raw))
    # depthMap = imadjust(depthMap);  % Enhance contrast if necessary
    # rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    # print(rgbd_image)
    DisplayImage(depth_raw, colormap=None)
    # print(np.asarray(depth_raw))
    pcl = open3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsics)
    # pcl = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics, extrinsic=extrinsics)
    print(open3d.io.write_point_cloud("cloud%s.pcd"%i, pcl, write_ascii=True, compressed=False, print_progress=True))
    vis.add_geometry(pcl)
    vis.run()
    vis.destroy_window()
