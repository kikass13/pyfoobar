
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d
import time
import sys


file = sys.argv[1]

# Read point cloud:
pcd = o3d.io.read_point_cloud(file)

downpcd = pcd.voxel_down_sample(voxel_size=0.1)

print(o3d.io.write_point_cloud("downsample_%s"%file, downpcd, write_ascii=True, compressed=False, print_progress=True))

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 1024, height = 768)
vis.add_geometry(downpcd)
vis.run()
vis.destroy_window()
