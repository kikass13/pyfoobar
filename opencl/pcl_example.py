from gpu_voxelizer import GpuVoxelizer
import numpy as np
import time
import copy

import matplotlib.pyplot as plt

def voxelize_cpu(points, T, VOXEL_SIZE, GRID_SIZE):
    T = T.astype(np.float32).flatten()
    grid_x, grid_y, grid_z = GRID_SIZE
    origin = -0.5 * np.array([grid_x, grid_y, grid_z], dtype=np.float32) * VOXEL_SIZE
    voxel_flags = np.zeros(grid_x * grid_y * grid_z)
    print(T)
    for p in points:
        x = T[0] * p[0] + T[1] * p[1] + T[2] * p[2] + T[4]
        y = T[4] * p[0] + T[5] * p[1] + T[6] * p[2] + T[7]
        z = T[8] * p[0] + T[9] * p[1] + T[10] * p[2] + T[11]
        vx = int(np.floor((x - origin[0]) / VOXEL_SIZE))
        vy = int(np.floor((y - origin[1]) / VOXEL_SIZE))
        vz = int(np.floor((z - origin[2]) / VOXEL_SIZE))
        if vx < 0 or vy < 0 or vz < 0 or vx >= grid_x or vy >= grid_y or vz >= grid_z:
            continue
        idx = vx + vy * grid_x + vz * grid_x * grid_y;
        voxel_flags[idx] = 1
    #####
    occupied = np.flatnonzero(voxel_flags)
    x = occupied % grid_x
    y = (occupied // grid_x) % grid_y
    z = occupied // (grid_x * grid_y)
    return np.stack([x, y, z], axis=1, dtype=np.float32), origin, voxel_size

######################################################################################################
### create voxelizer with centered grid and custom voxel size
grid_size = (25, 25, 10)
voxel_size = 1.0
voxelizer = GpuVoxelizer(grid_size=(grid_size), voxel_size=voxel_size)

for i in range(0, 3):
    N = 2_000_000
    t = np.linspace(0, 1, N)  # t goes from 0 to 1
    ###load or generate point clouds (example with random clouds)
    cloud1 = (1 - t)[:, None] * np.array([-10.0,-10.0,-10.0]) + t[:, None] * np.array([10.0,10.0,10.0])
    # cloud1 = np.random.rand(N, 3).astype(np.float32) * 40 - 20
    cloud2 = cloud1 + np.array([2.0, -1.0, 0.5])  # Offset version

    ### identity transforms for simplicity
    T = np.eye(4, dtype=np.float32)

    ### voxelize both clouds (separately)
    t0 = time.time()
    t = time.time()
    voxelizer.voxelize_cloud(cloud1, T)
    print(f"dt1: {time.time() - t}")
    t = time.time()
    voxelizer.voxelize_cloud(cloud2, T)
    # voxels, origin, voxel_size = voxelize_cpu(cloud2, T, voxel_size, grid_size)
    print(f"dt2: {time.time() - t}")
    ### extract voxel coordinates of occupied voxels
    t = time.time()
    voxels, origin, voxel_size = voxelizer.get_occupied_voxels()
    print(f"dtVoxels: {time.time() - t}")
    ### reset voxel grid after processing
    t = time.time()
    voxelizer.reset_voxel_grid()
    print(f"dtReset: {time.time() - t}")
    print(f"total dt: {time.time() - t0}")

#### plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Voxel Grid (red) and Point Cloud (blue)")
### plot original point cloud (downsampled for speed)
sampleN = int(N / 1000)
sample = cloud1[np.random.choice(len(cloud1), size=sampleN, replace=False)]
ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='blue', s=0.1, alpha=0.25)
sample2 = cloud2[np.random.choice(len(cloud2), size=sampleN, replace=False)]
ax.scatter(sample2[:, 0], sample2[:, 1], sample2[:, 2], c='purple', s=0.1, alpha=0.25)
### plot occupied voxels
voxel_coords = voxels * voxel_size + origin  # convert to world space
print(f"voxel_coords: {voxel_coords.shape} [{voxel_coords.dtype}]")
ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], c='red', s=10, alpha=1.0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()