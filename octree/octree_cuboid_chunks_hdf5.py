import pcl
import numpy as np
import os
import math
import h5py
import sys
import time

def split_into_octree_hdf5(pcd_file, hdf5_filename, chunk_size=10.0, octree_resolution=1.0):
    """
    Splits a large point cloud into cube chunks and stores them in a single HDF5 file.
    """
    print("loading point cloud...")
    cloud = pcl.load(pcd_file)
    ### convert to NumPy array for faster processing
    points = np.array(cloud)
    print(f"cloud points: {len(points)}")
    ### determine grid bounds
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    print(f"cloud boundaries: min={min_coords}, max={max_coords}")
    ### compute chunk grid dimensions
    num_x = int(np.ceil((max_coords[0] - min_coords[0]) / chunk_size))
    num_y = int(np.ceil((max_coords[1] - min_coords[1]) / chunk_size))
    num_z = int(np.ceil((max_coords[2] - min_coords[2]) / chunk_size))
    ### build an octree for fast radius-based search
    print("Building octree...")
    octree = cloud.make_octreeSearch(octree_resolution)
    octree.add_points_from_input_cloud()
    ### compute search radius (hypotenuse of half the cube)
    search_radius = (math.sqrt(3) / 2) * chunk_size
    print(f"Using radius search of {search_radius:.2f}m for pre-filtering")
    print(f"Number of expected chunks: x={num_x}, y={num_y}, z={num_z}")
    pointcount = 0
    ### open HDF5 file for writing
    with h5py.File(hdf5_filename, "w") as hdf5_file:
        ### create a dataset group in the file
        metadata_group = hdf5_file.create_group('metadata')
        metadata_group.create_dataset('description', data='chunks containing point cloud data')
        metadata_group.create_dataset('schema', data='x$_y$_z$')
        metadata_group.create_dataset('resolution', data=chunk_size)
        metadata_group.create_dataset("min_bounds", data=min_coords)
        metadata_group.create_dataset("max_bounds", data=max_coords)
        metadata_group.create_dataset("center", data=(max_coords + min_coords) / 2.0)
        metadata_group.create_dataset('num_chunks_x', data=num_x)
        metadata_group.create_dataset('num_chunks_y', data=num_y)
        metadata_group.create_dataset('num_chunks_z', data=num_z)
        metadata_group.create_dataset('stamp', data=time.time())
        chunk_data_group = hdf5_file.create_group('chunks')
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    ### define cube center
                    cube_center = min_coords + np.array([(i + 0.5) * chunk_size, 
                                                         (j + 0.5) * chunk_size, 
                                                         (k + 0.5) * chunk_size])
                    ### radius search (fast pre-filtering)
                    indices,_ = octree.radius_search(cube_center, search_radius)
                    ### extract exact cube from pre-filtered points
                    cube_min = min_coords + np.array([i, j, k]) * chunk_size
                    cube_max = cube_min + chunk_size
                    ### get points from radius search and filter them into the exact cube
                    extracted_points = points[indices]
                    in_cube = np.all((extracted_points >= cube_min) & (extracted_points < cube_max), axis=1)
                    final_cube_points = extracted_points[in_cube]
                    pointcount += len(final_cube_points)
                    ### store the chunk in HDF5
                    chunk_name = f"x{i}_y{j}_z{k}"
                    chunk_data_group.create_dataset(chunk_name, data=final_cube_points, compression="gzip")
                    print(f"Stored {chunk_name} with {len(final_cube_points)} points.")
    print("Octree-based cube chunking and HDF5 storage complete!")
    print(f"Chunkified point count = {pointcount}")

split_into_octree_hdf5(sys.argv[1], "octree_chunks.h5", chunk_size=10.0, octree_resolution=1.0)
