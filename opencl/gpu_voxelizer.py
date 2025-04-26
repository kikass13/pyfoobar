import numpy as np
import pyopencl as cl

class GpuVoxelizer:
    def __init__(self, grid_size=(128, 128, 128), voxel_size=0.1, origin=None, device_id=0, platform_id=0):
        self.GRID_X, self.GRID_Y, self.GRID_Z = grid_size
        self.VOXEL_SIZE = np.float32(voxel_size)
        if origin is None:
            self.ORIGIN = -0.5 * np.array([self.GRID_X, self.GRID_Y, self.GRID_Z], dtype=np.float32) * self.VOXEL_SIZE
        else:
            self.ORIGIN = np.array(origin, dtype=np.float32)

        self.platform = cl.get_platforms()[platform_id]
        self.device = self.platform.get_devices()[device_id]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.mf = cl.mem_flags

        self.total_voxels = self.GRID_X * self.GRID_Y * self.GRID_Z
        self.voxel_flags = np.zeros(self.total_voxels, dtype=np.uint8)
        self.voxel_flags_buf = cl.Buffer(self.context, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.voxel_flags)

        self.program = self._build_kernel()

    def _build_kernel(self):
        kernel_code = f"""
        #define GRID_X {self.GRID_X}
        #define GRID_Y {self.GRID_Y}
        #define GRID_Z {self.GRID_Z}

        __kernel void transform_voxelize(
            __global const float* points,
            __global const float* T,
            int num_points,
            float voxel_size,
            float3 origin,
            __global char* voxel_flags
        ) {{
            int gid = get_global_id(0);
            if (gid >= num_points) return;

            float px = points[gid * 3];      // X coordinate
            float py = points[gid * 3 + 1];  // Y coordinate
            float pz = points[gid * 3 + 2];  // Z coordinate

            float x = T[0] * px + T[1] * py + T[2] * pz + T[3];
            float y = T[4] * px + T[5] * py + T[6] * pz + T[7];
            float z = T[8] * px + T[9] * py + T[10] * pz + T[11];

            int vx = (int)floor((x - origin.x) / voxel_size);
            int vy = (int)floor((y - origin.y) / voxel_size);
            int vz = (int)floor((z - origin.z) / voxel_size);

            if (vx < 0 || vy < 0 || vz < 0 || vx >= GRID_X || vy >= GRID_Y || vz >= GRID_Z)
                return;

            int grid_index = vx + (vy * GRID_X) + (vz * GRID_X * GRID_Y);
            voxel_flags[grid_index] = 1;
        }}
        """
        return cl.Program(self.context, kernel_code).build()

    def voxelize_cloud(self, points: np.ndarray, transform: np.ndarray):
        """Voxelize a single cloud using 3D points."""
        assert points.shape[1] == 3
        assert transform.shape == (4, 4)

        points3 = points.astype(np.float32)
        points_buf = cl.Buffer(self.context, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=points3)

        transform_flatten = transform.astype(np.float32).flatten()
        transform_buf = cl.Buffer(self.context, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=transform_flatten)

        event = self.program.transform_voxelize(
            self.queue, (points.shape[0],), None,
            points_buf, transform_buf,
            np.int32(points.shape[0]),
            self.VOXEL_SIZE, self.ORIGIN, self.voxel_flags_buf
        )
        return event

    def get_occupied_voxels(self):
        cl.enqueue_copy(self.queue, self.voxel_flags, self.voxel_flags_buf)
        occupied = np.flatnonzero(self.voxel_flags)
        x = occupied % self.GRID_X
        y = (occupied // self.GRID_X) % self.GRID_Y
        z = occupied // (self.GRID_X * self.GRID_Y)
        world_coords = np.stack([x, y, z], axis=1, dtype=np.float32)
        return world_coords, self.ORIGIN, self.VOXEL_SIZE

    def reset_voxel_grid(self):
        self.voxel_flags.fill(0)
        cl.enqueue_copy(self.queue, self.voxel_flags_buf, self.voxel_flags)
