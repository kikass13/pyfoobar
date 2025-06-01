import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import cv2
import time

MIN_EXPECTED_OBJECT_SPEED = 0.0
MAX_EXPECTED_OBJECT_SPEED = 30.0

APPROXIMATE_CIRCLE = False

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
    return np.clip(grid / (grid.max() + 1e-5), 0, 1)

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

def process_object(n_steps, object_grid, obj : MyObject, cfg : MyConfig):
    cx,cy = obj.position
    vx,vy = obj.velocity
    ax,ay = obj.acceleration
    
    dt = compute_deceleration_dt(vx, vy)
    dt *= cfg.DT_LOOKAHEAD_FACTOR

    first = np.zeros_like(object_grid)
    for step in range(n_steps):
        stepT = time.time()
        t = step / (n_steps - 1)
        elapsed_time, cost, newPos, newVel, newAcc, newYaw, newSize = calculate_object_step_parameter(t, dt, obj.position, obj.velocity, obj.acceleration, obj.size, cfg.DT_FOOTPRINT_SCALE_FACTOR )
        temp = np.zeros_like(object_grid, dtype=np.uint8)
        yaw_deg = np.degrees(newYaw)
        if APPROXIMATE_CIRCLE:
            ix, iy = (int(newPos[0]), int(newPos[1]))
            if 0 <= ix < object_grid.shape[1] and 0 <= iy < object_grid.shape[0]:
                cv2.circle(temp, (ix,iy), radius=int(np.ceil(newSize[0])), color=255, thickness=-1)
        else:
            draw_rotated_rectangle(temp, (newPos[0], newPos[1]), (newSize[0], newSize[1]), yaw_deg, value=255)
        sigma = compute_adaptive_speed_to_X(obj.speed, cfg.min_sigma, cfg.max_sigma)
        blurred = cv2.GaussianBlur(temp.astype(np.float32), cfg.blur_kernel, sigmaX=sigma)
        object_grid += blurred / 255.0 * cost
        ### remember first result as mask for later
        if step == 0:
            first = blurred.copy()
    return object_grid, first

def main():
    # Grid size
    resolution = 1.0
    bounds = np.array([100, 50])
    
    # H, W = 50, 25
    GRID_SIZE = bounds / resolution
    H, W = (int(GRID_SIZE[0]), int(GRID_SIZE[1]))
    grid = np.zeros((H, W), dtype=np.float32)

    # Object parameters
    n_objects = 30
    np.random.seed(0)
    positions = np.random.uniform(0, H, size=(n_objects, 2))
    angles = np.random.uniform(0, 2 * np.pi, size=n_objects)
    speeds = np.random.uniform(0, 30, size=n_objects)
    accelerations = np.random.uniform(-2, 2, size=(n_objects, 2))  # ax, ay per object

    n_min_steps = 2
    n_max_steps = 50
    DT_LOOKAHEAD_FACTOR = 2.0
    DT_FOOTPRINT_SCALE_FACTOR = 1.0 ### dynamically makes the footprint bigger
    # blur_kernel = (21, 21)
    blur_kernel = (5, 5) ### faster but less big
    MIN_SIGMA = 2.0
    MAX_SIGMA = 5.0

    BASE_WIDTH = 1.5
    BASE_LENGTH = 3.0

    start_points = []
    vel_vectors = []
    acc_vectors = []

    timesStep = []
    timesDraw = []
    timesGauss = []

    first_grids = []

    processing_t0 = time.time()
    for i in range(n_objects):
        cx, cy = positions[i]
        vx, vy = pol2cart(speeds[i], angles[i])
        ax, ay = accelerations[i]
        ### constant velocity
        # yaw_rad = np.arctan2(vy, vx)
        # yaw_deg = np.degrees(yaw_rad)

        start_points.append((cx, cy))
        vel_vectors.append((vx, vy))
        acc_vectors.append((ax, ay))

        object_grid = np.zeros((H, W), dtype=np.float32)
        obj = MyObject(np.array([cx, cy]), np.array([vx,vy]), np.array([ax,ay]), np.array([BASE_LENGTH, BASE_WIDTH]))
        cfg = MyConfig(blur_kernel, MIN_SIGMA, MAX_SIGMA, DT_LOOKAHEAD_FACTOR, DT_FOOTPRINT_SCALE_FACTOR=DT_FOOTPRINT_SCALE_FACTOR)
        n_steps = compute_adaptive_speed_to_X(obj.speed, n_min_steps, n_max_steps)
        object_grid, first = process_object(n_steps, object_grid, obj, cfg)
        first_grids.append(first)
        # Normalize the object grid to ensure max value is 1.0
        object_grid = normalize_grid(object_grid)

        # Merge this object grid into the global grid
        # grid += object_grid
        # Merge this object grid into the global grid and clip the values to avoid oversaturation
        grid = np.clip(grid + object_grid, 0, 1)

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
    # dynamic_mask = grid > 0.0 ### compelte path
    dynamic_mask_result = np.zeros_like(grid)
    for mask in first_grids:
        dynamic_mask_result += mask ### still a cost image, add all first layers together
    dynamic_mask_result = dynamic_mask_result > 0.0 ### now a boolean image
    plt.imshow(dynamic_mask_result, origin='lower', cmap='cool', extent=[0, W, 0, H], alpha=0.3)
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
    previous_layer_grid[dynamic_mask_result] = 0  # or *= 0.5 for soft cut
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
