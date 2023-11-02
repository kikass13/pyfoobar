import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d
import time
import sys
import os

file = sys.argv[1]

start = time.time()

# Read point cloud:
pcd = o3d.io.read_point_cloud(file)

# Get points and transform it to a numpy array:
xyz = np.asarray(pcd.points, dtype=np.float32)
rgb = np.asarray(pcd.colors, dtype=np.float32)
end = time.time()
print(end-start)

# Combine XYZ and RGB arrays into a single XYZRGB array
xyzrgb = np.concatenate((xyz, rgb), axis=1)
print(xyzrgb)

### pickle the points and load again for time comparison
filename = os.path.splitext(file)[0]
print("Writing Numpy Pickle '%s'" % filename)
np.save(filename, xyzrgb)

start = time.time()
points = np.load(filename+".npy", allow_pickle=True)
end = time.time()
print(end-start)




vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 1024, height = 768)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
