#define GRID_X {self.GRID_X}
#define GRID_Y {self.GRID_Y}
#define GRID_Z {self.GRID_Z}

__kernel void transform_voxelize(
    __global const float4* points,
    __global const float16* transform,
    int num_points,
    float voxel_size,
    float3 origin,
    __global char* voxel_flags
) {
    int gid = get_global_id(0);
    if (gid >= num_points) return;

    float4 p = points[gid];
    float16 T = transform[0];

    float x = T.s0 * p.x + T.s1 * p.y + T.s2 * p.z + T.s3;
    float y = T.s4 * p.x + T.s5 * p.y + T.s6 * p.z + T.s7;
    float z = T.s8 * p.x + T.s9 * p.y + T.sa * p.z + T.sb;

    int vx = (int)floor((x - origin.x) / voxel_size);
    int vy = (int)floor((y - origin.y) / voxel_size);
    int vz = (int)floor((z - origin.z) / voxel_size);

    if (vx < 0 || vy < 0 || vz < 0 || vx >= GRID_X || vy >= GRID_Y || vz >= GRID_Z)
        return;

    int idx = vx + vy * GRID_X + vz * GRID_X * GRID_Y;
    voxel_flags[idx] = 1;
}