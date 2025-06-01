from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import cv2
import time

MIN_EXPECTED_OBJECT_SPEED = 0.0
MAX_EXPECTED_OBJECT_SPEED = 30.0

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def compute_deceleration_dt(vx, vy, decel=6.0):
    speed = np.hypot(vx, vy)
    dt = speed / decel
    return dt

def smooth_triangle(t, sharpness=1.0, min_cost=0.0):
    if t <= 0.5:
        cost = (t / 0.5) ** sharpness
    else:
        cost = ((1 - t) / 0.5) ** sharpness
    return min_cost + (1.0 - min_cost) * cost

def cost_distribution_based_on_elapsed_time(t, sharpness=1.0, min_cost=0.0):
    return smooth_triangle(t, sharpness=sharpness, min_cost=min_cost)

def draw_rotated_rectangle(image, center, size, angle_deg, value=1.0):
    box = cv2.boxPoints((center, size, angle_deg))
    box = np.intp(box)
    cv2.fillConvexPoly(image, box, color=value)

def normalize_grid(grid):
    """Normalize the grid so that its maximum value is 1.0."""
    return np.clip(grid / grid.max(), 0, 1)


class MyObject:
    def __init__(self, position, velocity, acceleration, size ):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.size = size #### length / width
        self.speed = np.linalg.norm(velocity)

def compute_adaptive_speed_to_X(speed, minV, maxV):
    return int(np.clip(np.interp(speed, [MIN_EXPECTED_OBJECT_SPEED, MAX_EXPECTED_OBJECT_SPEED], [minV, maxV]), minV, maxV))

class MyConfig:
    def __init__(self, blur_kernel, min_sigma, max_sigma, DT_LOOKAHEAD_FACTOR, DT_FOOTPRINT_SCALE_FACTOR=1.0):
        self.blur_kernel = blur_kernel
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.DT_LOOKAHEAD_FACTOR = DT_LOOKAHEAD_FACTOR
        self.DT_FOOTPRINT_SCALE_FACTOR = DT_FOOTPRINT_SCALE_FACTOR

@nb.njit()
def calculate_object_step_parameter(t, dt, position, velocity, acceleration, size, DT_FOOTPRINT_SCALE_FACTOR):
    elapsed_time = t * dt
    ### constant velocity
    # x = cx + vx * elapsed_time
    # y = cy + vy * elapsed_time
    ### constant acceleration
    pos_t = position + velocity * elapsed_time + 0.5 * acceleration * elapsed_time**2
    vel_t = velocity + acceleration * elapsed_time
    acc_t = acceleration
    yaw = np.arctan2(vel_t[1], vel_t[0])
    speed = np.linalg.norm(vel_t)
    #############
    # cost = cost_distribution_based_on_elapsed_time(t, sharpness=0.5, min_cost=0.1)
    min_cost = 0.1
    sharpness = 0.5
    if t <= 0.5:
        cost = (t / 0.5) ** sharpness
    else:
        cost = ((1 - t) / 0.5) ** sharpness
    cost = min_cost + (1.0 - min_cost) * cost
    #############
    scale_factor = DT_FOOTPRINT_SCALE_FACTOR * t
    length_scaled = size[0] * (1.0 + 0.05 * speed) * scale_factor
    width_scaled = size[1] * (1.0 + 0.1 * speed) * scale_factor
    return elapsed_time, cost, pos_t, vel_t, acc_t, yaw, np.array([length_scaled, width_scaled])
calculate_object_step_parameter(0.0, 0.0, np.zeros((2),dtype=np.float64), np.zeros((2),dtype=np.float64), np.zeros((2), dtype=np.float64), np.zeros((2), dtype=np.float64), 1.0) ### numba precomp

def process_object(i, H, W, positions, angles, speeds, accelerations, min_steps, max_steps,
                   DT_LOOKAHEAD_FACTOR,
                   BASE_LENGTH, BASE_WIDTH, blur_kernel, MIN_SIGMA, MAX_SIGMA):

    cx, cy = positions[i]
    vx, vy = pol2cart(speeds[i], angles[i])
    ax, ay = accelerations[i]
    speed = np.hypot(vx, vy)

    dt = compute_deceleration_dt(vx, vy) * DT_LOOKAHEAD_FACTOR
    object_grid = np.zeros((H, W), dtype=np.float32)

    n_steps = compute_adaptive_speed_to_X(speed, min_steps, max_steps)
    for step in range(n_steps):
        t = step / (n_steps - 1)
        elapsed_time = t * dt
        x = cx + vx * elapsed_time + 0.5 * ax * elapsed_time ** 2
        y = cy + vy * elapsed_time + 0.5 * ay * elapsed_time ** 2
        vx_t = vx + ax * elapsed_time
        vy_t = vy + ay * elapsed_time
        yaw_deg = np.degrees(np.arctan2(vy_t, vx_t))
        cost = cost_distribution_based_on_elapsed_time(t, sharpness=0.5, min_cost=0.1)

        # scale_factor = 1.0 + (DT_FOOTPRINT_SCALE_FACTOR * t)
        # speed = np.hypot(vx, vy)
        # length = BASE_LENGTH * (1.0 + 0.05 * speed) * scale_factor
        # width = BASE_WIDTH * (1.0 + 0.1 * speed) * scale_factor
        ###
        length = BASE_LENGTH
        width = BASE_WIDTH
        ###
        temp = np.zeros_like(object_grid, dtype=np.uint8)
        draw_rotated_rectangle(temp, (x, y), (length, width), yaw_deg, value=255)
        SIGMA = compute_adaptive_speed_to_X(speed, MIN_SIGMA, MAX_SIGMA)
        blurred = cv2.GaussianBlur(temp.astype(np.float32), blur_kernel, sigmaX=SIGMA)
        object_grid += blurred / 255.0 * cost

    return normalize_grid(object_grid), (cx, cy), (vx, vy), (ax, ay)

def main():
    # Grid size
    H, W = 200, 200
    grid = np.zeros((H, W), dtype=np.float32)

    # Object parameters
    n_objects = 30
    np.random.seed(0)
    positions = np.random.uniform(25, 175, size=(n_objects, 2))
    angles = np.random.uniform(0, 2 * np.pi, size=n_objects)
    speeds = np.random.uniform(0, 30, size=n_objects)
    accelerations = np.random.uniform(-2, 2, size=(n_objects, 2))  # ax, ay per object

    min_steps = 2
    max_steps = 50
    DT_LOOKAHEAD_FACTOR = 2.0
    # blur_kernel = (21, 21)
    blur_kernel = (5, 5) ### faster but less smooth
    MIN_SIGMA = 1.0
    MAX_SIGMA = 10.0

    BASE_WIDTH = 2.0
    BASE_LENGTH = 4.5

    start_points = []
    vel_vectors = []
    acc_vectors = []

    timesStep = []
    timesDraw = []
    timesGauss = []

    processing_t0 = time.time()
    # with ThreadPoolExecutor() as executor:
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_object,
                i, H, W, positions, angles, speeds, accelerations,
                min_steps, max_steps, DT_LOOKAHEAD_FACTOR,
                BASE_LENGTH, BASE_WIDTH, blur_kernel, MIN_SIGMA, MAX_SIGMA
            )
            for i in range(n_objects)
        ]

        for future in futures:
            object_grid, start_pt, vel_vec, acc_vec = future.result()
            grid = np.clip(grid + object_grid, 0, 1)
            start_points.append(start_pt)
            vel_vectors.append(vel_vec)
            acc_vectors.append(acc_vec)

    # Normalize the final grid so it doesn't exceed 1.0
    # grid = np.clip(grid / grid.max(), 0, 1)
    # print(f"draw time mean: {np.mean(timesDraw)} s")
    # print(f"gauss time mean: {np.mean(timesGauss)} s")
    # print(f"Step time mean: {np.mean(timesStep)} s")
    # print(f"Step time sum: {np.sum(timesStep)} s")
    print(f"Processing time: {time.time() - processing_t0:.3f} s")
    # Plotting
    plt.figure(figsize=(7, 7))
    plt.imshow(grid, origin='lower', cmap='hot', extent=[0, W, 0, H], alpha=0.7)
    plt.colorbar(label="Motion Cost")
    plt.title("Merged Rotated Rectangle Costmap with Velocity Vectors")
    plt.grid(alpha=0.3)

    for i, ((cx, cy), (vx, vy), (ax, ay)) in enumerate(zip(start_points, vel_vectors, acc_vectors)):
        yaw_rad = np.arctan2(vy, vx)
        yaw_deg = np.degrees(yaw_rad)
        box = cv2.boxPoints(((cx, cy), (BASE_LENGTH, BASE_WIDTH), yaw_deg))
        box = np.array(box)
        plt.plot(cx, cy, 'go', markersize=8)
        plt.plot(*np.append(box, [box[0]], axis=0).T, color='lime', linewidth=2)
        plt.arrow(cx, cy, vx * 0.5, vy * 0.5, head_width=1.5, head_length=2, fc='cyan', ec='cyan')
        plt.arrow(cx, cy, ax , ay, head_width=1.0, head_length=1.0, fc='blue', ec='blue')
    ### mask for dynamic objects
    DYNAMIC_OBJECT_THRESHOLD = 0.01
    dynamic_mask = grid > DYNAMIC_OBJECT_THRESHOLD
    plt.imshow(dynamic_mask, origin='lower', cmap='cool', extent=[0, W, 0, H], alpha=0.3)
    ###
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.tight_layout()
    plt.show()

    ############## DEBUG
    ########
    ### create a sensor cost map before, where we mask our additions and can see if we would mask out dynamic objects from the 
    ### sensor layer
    ### for demonstration: simulate static LIDAR layer
    previous_layer_grid = np.random.uniform(0, 1, size=(H, W)).astype(np.float32)
    previous_layer_grid = np.clip(previous_layer_grid + grid, 0, 1)  # add fake lidar + dynamic clutter
    previous_layer_grid[dynamic_mask] = 0  # or *= 0.5 for soft cut
    plt.figure(figsize=(6, 6))
    plt.imshow(previous_layer_grid, origin='lower', cmap='viridis', extent=[0, W, 0, H])
    plt.title("Static Grid After Suppression of Dynamic Objects")
    plt.grid(alpha=0.3)
    plt.colorbar(label='Cost')
    plt.tight_layout()
    plt.show()
    ########
    ############## DEBUG


if __name__ == "__main__":
    main()
