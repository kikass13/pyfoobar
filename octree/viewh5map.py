import sys
import h5py
import numpy as np
import open3d as o3d
import time

def viz_pcl(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    o3d.visualization.draw_geometries([pcd])

###visualize everything
def load_and_viz_all():
    all_points = []
    with h5py.File(sys.argv[1], 'r') as f:
        meta = f["metadata"]
        for member in meta:
            print(f"{member}: {meta[member][()]}")
        chunks = f["chunks"]
        for chunk_name in chunks:
            points = np.array(chunks[chunk_name][:])
            all_points.extend(points)

    all_points = np.array(all_points)
    viz_pcl(all_points)

def get_chunk_name(i, j, k):
    return f"x{i}_y{j}_z{k}"
def get_chunk_indices(position, chunk_size):
    ### calculate the chunk indices for each dimension based on the position and chunk size
    i = int(np.floor(position[0] / chunk_size))
    j = int(np.floor(position[1] / chunk_size))
    k = int(np.floor(position[2] / chunk_size))
    return i, j, k
def get_relevant_chunk_names(position, range_radius, chunk_size, global_offset):
    local_position = position - global_offset
    ### calculate the min and max coordinates for the area to load
    min_x = local_position[0] - range_radius
    min_y = local_position[1] - range_radius
    min_z = local_position[2] - range_radius
    max_x = local_position[0] + range_radius
    max_y = local_position[1] + range_radius
    max_z = local_position[2] + range_radius
    ### calculate chunk indices for the min and max coordinates
    min_chunk = get_chunk_indices((min_x, min_y, min_z), chunk_size)
    max_chunk = get_chunk_indices((max_x, max_y, max_z), chunk_size)
    ### generate chunk names within the range
    chunk_names = []
    for i in range(min_chunk[0], max_chunk[0] + 1):
        for j in range(min_chunk[1], max_chunk[1] + 1):
            for k in range(min_chunk[2], max_chunk[2] + 1):
                chunk_names.append(get_chunk_name(i, j, k))
    return chunk_names

### only load and visualize the chunks near specified position
def load_and_viz_specific(range_radius, position=None):
    all_points = []
    with h5py.File(sys.argv[1], 'r') as f:
        meta = f["metadata"]
        chunk_size = meta["resolution"][()]
        global_offset = meta["min_bounds"][()]
        center = meta["center"][()]
        if position == None:
            position = center
        chunk_names = get_relevant_chunk_names(position, range_radius, chunk_size, global_offset)
        chunks = f["chunks"]
        for chunk_name in chunk_names:
            if chunk_name in chunks:
                all_points.extend(chunks[chunk_name][:])
    return all_points


# load_and_viz_all()
range_radius = 100.0
t = time.time()
points = load_and_viz_specific(range_radius, position=None) ## center of map
print(f"chunk loading dt: {time.time() - t}")
viz_pcl(points)
# position = np.array([3.0511466e+05, 5.6434975e+06, 0.3])
# load_and_viz_specific(range_radius, position=position)