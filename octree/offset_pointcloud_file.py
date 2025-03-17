import sys
import os
import numpy as np

import open3d as o3d

file = sys.argv[1]

# Read point cloud:
pcd = o3d.io.read_point_cloud(file)

OFFSET = np.array([0.0,0.0,118.5])
x,y,z = OFFSET.tolist()
offsetStr = f'offset_x{str(x).replace(".", "_")}y{str(y).replace(".", "_")}z{str(z).replace(".", "_")}'
fileWoExtension, extension =  os.path.splitext(file)
strName = f"{fileWoExtension}_{offsetStr}{extension}"

offset_points = pcd.points - OFFSET
pcd.points = o3d.utility.Vector3dVector(offset_points)
print(o3d.io.write_point_cloud(strName, pcd, write_ascii=True, compressed=False, print_progress=True))

