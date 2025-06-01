import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import time
from scipy.ndimage import binary_dilation

# --- OpenCL kernel: Draw cost for rotated rectangle with 5x5 Gaussian blur
opencl_kernel_code = """

#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define KERNEL_SUM 256

#define DT_0_COST_MULTIPLIER 1.5
#define DT_1_COST_MULTIPLIER 3.0
#define DT_2_COST_MULTIPLIER 1.0

inline float compute_deceleration_dt(float2 v, float decel) {
    float speed = sqrt(v[0] * v[0] + v[1] * v[1]);
    return speed / decel;
}

__kernel void process_objects(
    __global const float2* positions,
    __global const float2* velocities,
    __global const float2* accelerations,
    __global const float2* sizes,
    const float DT_LOOKAHEAD_FACTOR,
    const float DT_EXPECTED_OBJECT_DECELERATION,
    const float OBJECT_DYNAMIC_SCALE_FACTOR,
    const float COST_DISTRIBUTION_SHARPNESS,
    const float COST_DISTRIBUTION_MIN_COST,
    const int steps,
    const int image_height,
    const int image_width,
    __global int* gaussian_kernel,
    __global int* grid_out,
    __global int* mask_grid
) {
    int obj_id = get_global_id(0);
    int step_id = get_global_id(1);

    float2 pos0 = positions[obj_id];
    float2 vel0 = velocities[obj_id];
    float2 acc0 = accelerations[obj_id];

    // t defines, how much time of our dt lookahead window has passed (in %, 0 to 1)
    float t = (float)step_id / (steps - 1);

    /// calculate dt adaptiveley, based on objects current (start) velocity 
    float dt = compute_deceleration_dt(vel0, DT_EXPECTED_OBJECT_DECELERATION);
    float elapsed = t * dt;

    float2 pos = pos0 + vel0 * elapsed + 0.5f * acc0 * elapsed * elapsed;
    float2 vel = vel0 + acc0 * elapsed;
    float speed = sqrt(vel.x * vel.x + vel.y * vel.y);
    float yaw = atan2(vel.y, vel.x);

    float2 size = sizes[obj_id];
    float length = size.x * (1.0f + 0.05f * speed) * t * OBJECT_DYNAMIC_SCALE_FACTOR;
    float width = size.y * (1.0f + 0.10f * speed) * t * OBJECT_DYNAMIC_SCALE_FACTOR;

    // tringle function with given sharpness and min cost val
    float cost = 0.0f;
    /// make the cost relativeley low at the dt x 0.5 mark
    if(t <= 0.4f){
        cost = pow(t / 0.5f, COST_DISTRIBUTION_SHARPNESS) * DT_0_COST_MULTIPLIER;
    }
    /// make the cost atrificially high at around the 1xdt mark
    else if(t > 0.4 && t < 0.6f){
        cost = pow(t / 0.5f, COST_DISTRIBUTION_SHARPNESS) * DT_1_COST_MULTIPLIER;
    }
    else{
        cost = pow((1.0f - t) / 0.5f, COST_DISTRIBUTION_SHARPNESS) * DT_2_COST_MULTIPLIER;
    }
    cost = COST_DISTRIBUTION_MIN_COST + (1.0f - COST_DISTRIBUTION_MIN_COST) * cost;
    int cost_i = (int)(cost * 1000.0f);

    int rad_x = (int)(width / 2);
    int rad_y = (int)(length / 2);

    // printf("costi: %d", cost_i);

    for (int dy = -rad_y; dy <= rad_y; dy++) {
        for (int dx = -rad_x; dx <= rad_x; dx++) {
            float fx = (float)dx;
            float fy = (float)dy;

            float rx = fx * cos(yaw) - fy * sin(yaw);
            float ry = fx * sin(yaw) + fy * cos(yaw);

            int px = (int)(pos.x + rx);
            int py = (int)(pos.y + ry);

            if (px >= 0 && px < image_width && py >= 0 && py < image_height) {
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                        int gx = px + kx;
                        int gy = py + ky;

                        if (gx >= 0 && gx < image_width && gy >= 0 && gy < image_height) {
                            int idx = gy * image_width + gx;
                            int k_idx = (ky + KERNEL_RADIUS) * (2 * KERNEL_RADIUS + 1) + (kx + KERNEL_RADIUS);
                            int weight = gaussian_kernel[k_idx];
                            int blurred_cost = (cost_i * weight) / KERNEL_SUM;
                            // printf("blurred_cost: %f", blurred_cost);
                            /// add overall cost
                            atomic_add(&grid_out[idx], blurred_cost);
                            /// first step (current objects position) goes into mask
                            if (step_id == 0) {
                                atomic_add(&mask_grid[idx], blurred_cost);
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


def log_normalize(grid, epsilon=0.1):
    # Apply a logarithmic transformation with a small constant to avoid log(0)
    transformed_grid = np.log(grid + epsilon)
    normalized_grid = (transformed_grid - transformed_grid.min()) / (transformed_grid.max() - transformed_grid.min())
    return normalized_grid
def smooth_compress(grid, gamma=0.5):
    # Similar to gamma correction: less aggressive than log
    grid = np.power(grid, gamma)
    return grid / np.max(grid)
def normalize(grid):
    return np.clip(grid / grid.max(), 0, 1)

def normalize_grid(grid, mode="normal", epsilon=1e-6, gamma=0.5):
    if mode == "log":
        return log_normalize(grid, epsilon=epsilon)
    elif mode == "smooth":
        return smooth_compress(grid, gamma)
    else:
        return normalize(grid)


# --- Parameters
resolution = 0.5
bounds = np.array([100, 100])
GRID_SIZE = bounds / resolution
H, W = (int(GRID_SIZE[0]), int(GRID_SIZE[1]))

N = 30      # number of objects
T = 50      # time steps
DT_LOOKAHEAD_FACTOR = 2.0
DT_EXPECTED_DECELERATION = 6.0 # m/s²
OBJECT_DYNAMIC_SCALE_FACTOR = 1.0
COST_DISTRIBUTION_SHARPNESS = 0.5
COST_DISTRIBUTION_MIN_COST = 0.1
np.random.seed(0)

# --- Data
positions = np.random.uniform([20, 20], [W-20, H-20], size=(N, 2)).astype(np.float32)
angles = np.random.uniform(0, 2*np.pi, size=N)
speeds = np.random.uniform(1, 30, size=N)
velocities = np.stack([np.cos(angles) * speeds, np.sin(angles) * speeds], axis=1).astype(np.float32)
accelerations = np.random.uniform(-2, 2, size=(N, 2)).astype(np.float32)
sizes = np.tile(np.array([[4.5, 2.0]], dtype=np.float32), (N, 1))

gaussian_kernel = np.array([
    1,  4,  6,  4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1,  4,  6,  4, 1
], dtype=np.int32)

# --- Setup OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, opencl_kernel_code).build()

for i in range(2):  # warm-up and measure
    t = time.time()
    mf = cl.mem_flags
    buf_pos = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions)
    buf_vel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=velocities)
    buf_acc = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=accelerations)
    buf_kernel = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gaussian_kernel)
    buf_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sizes)
    grid = np.zeros((H, W), dtype=np.int32)
    buf_grid = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=grid)
    mask_grid = np.zeros_like(grid)
    buf_mask = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mask_grid)
    prg.process_objects(
        queue, (N, T), None,
        buf_pos, buf_vel, buf_acc, buf_size,
        np.float32(DT_LOOKAHEAD_FACTOR), np.float32(DT_EXPECTED_DECELERATION), np.float32(OBJECT_DYNAMIC_SCALE_FACTOR),
        np.float32(COST_DISTRIBUTION_SHARPNESS), np.float32(COST_DISTRIBUTION_MIN_COST),
        np.int32(T), np.int32(H), np.int32(W),
        buf_kernel,
        buf_grid,
        buf_mask,
    )
    cl.enqueue_copy(queue, grid, buf_grid)
    cl.enqueue_copy(queue, mask_grid, buf_mask)
    # Normalize the object grid to ensure max value is 1.0
    grid = (grid.astype(np.float32) / 1000.0) ### make the grid float again
    # grid = normalize_grid(grid)
    # grid = normalize_grid(grid, mode="log", epsilon=0.1)
    grid = normalize_grid(grid, mode="smooth", gamma=0.5)

    # clip grid to avoid oversaturation
    # grid = np.clip(grid, 0, 10000)  # e.g., MAX_VALUE = 1000 or some meaningful threshold
    # for row in range(grid.shape[0]):
    #     print(grid[row])
    print("Frame", i, "rendered in", time.time() - t, "sec")

# --- Plot Results
plt.figure(figsize=(8, 7))
plt.imshow(grid, origin='lower', extent=[0, W, 0, H], cmap='hot', alpha=0.9)
plt.colorbar(label="Cost (normalized)")

for i in range(N):
    cx, cy = positions[i]
    vx, vy = velocities[i]
    yaw = np.arctan2(vy, vx)
    l, w = sizes[i]
    rect = np.array([
        [-l/2, -w/2],
        [-l/2,  w/2],
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2]
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rect = rect @ R.T + [cx, cy]
    plt.plot(rect[:, 0], rect[:, 1], 'lime', lw=1.5)
    plt.arrow(cx, cy, vx*0.3, vy*0.3, color='cyan', head_width=1.5)



tmask = time.time()
# dynamic_mask = grid > 0.0 ### compelte path
dynamic_mask_result = mask_grid > 0 ### now a boolean image
### inflate mask
MASK_INFLATIONS = 1
if MASK_INFLATIONS > 0:
    dynamic_mask_result = binary_dilation(dynamic_mask_result, structure=np.ones((3, 3)), iterations=MASK_INFLATIONS)
print(f"mask layer processing dt: {time.time()- tmask}")

plt.title("OpenCL GPU Costmap with 5×5 Gaussian Blur (Per Object-Step)")
plt.xlim(0, W)
plt.ylim(0, H)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

############## DEBUG
########
### create a sensor cost map before, where we mask our additions and can see if we would mask out dynamic objects from the 
### sensor layer
### for demonstration: simulate static LIDAR layer
previous_layer_grid = np.random.uniform(0, 1, size=(H, W)).astype(np.float32)
previous_layer_grid = np.clip(previous_layer_grid + grid, 0, 1)  # add fake lidar + dynamic clutter
previous_layer_grid[dynamic_mask_result] = 0  # or *= 0.5 for soft cut
plt.figure(figsize=(6, 6))
plt.imshow(previous_layer_grid, origin='lower', cmap='viridis', extent=[0, W, 0, H])
plt.imshow(dynamic_mask_result, origin='lower', cmap='cool', extent=[0, W, 0, H], alpha=0.3)
plt.title("Static Grid After Suppression of Dynamic Objects")
plt.grid(alpha=0.3)
plt.colorbar(label='Cost')
plt.tight_layout()
plt.show()
########
############## DEBUG
