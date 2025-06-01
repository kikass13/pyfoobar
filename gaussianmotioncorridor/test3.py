import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


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

def main():
    # Grid size
    H, W = 200, 200
    grid = np.zeros((H, W), dtype=np.float32)

    # Object parameters
    n_objects = 20
    np.random.seed(0)
    positions = np.random.uniform(25, 175, size=(n_objects, 2))
    angles = np.random.uniform(0, 2 * np.pi, size=n_objects)
    speeds = np.random.uniform(5, 10, size=n_objects)
    accelerations = np.random.uniform(-2, 2, size=(n_objects, 2))  # ax, ay per object

    n_steps = 5
    DT_LOOKAHEAD_FACTOR = 2.0
    DT_FOOTPRINT_SCALE_FACTOR = 0.0 ### dynamically makes the footprint bigger
    # blur_kernel = (21, 21)
    blur_kernel = (5, 5) ### faster but less smooth
    SIGMA = 2.0

    BASE_WIDTH = 2.0
    BASE_LENGTH = 4.5

    start_points = []
    vel_vectors = []
    acc_vectors = []

    timesStep = []
    timesDraw = []
    timesGauss = []
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

        dt = compute_deceleration_dt(vx, vy)
        dt *= DT_LOOKAHEAD_FACTOR

        object_grid = np.zeros((H, W), dtype=np.float32)

        for step in range(n_steps):
            processing_tstep = time.time()
            t = step / (n_steps - 1)
            elapsed_time = t * dt
            ### constant velocity
            # x = cx + vx * elapsed_time
            # y = cy + vy * elapsed_time
            ### constant acceleration
            x = cx + vx * elapsed_time + 0.5 * ax * elapsed_time ** 2
            y = cy + vy * elapsed_time + 0.5 * ay * elapsed_time ** 2
            vx_t = vx + ax * elapsed_time
            vy_t = vy + ay * elapsed_time
            yaw_rad = np.arctan2(vy_t, vx_t)
            yaw_deg = np.degrees(yaw_rad)
            ###
            cost = cost_distribution_based_on_elapsed_time(t, sharpness=0.5, min_cost=0.1)
            # Scale rectangle size with speed and t
            scale_factor = 1.0 + (DT_FOOTPRINT_SCALE_FACTOR * t)  # Gradually grows with time
            speed = np.hypot(vx, vy)
            length = BASE_LENGTH * (1.0 + 0.05 * speed) * scale_factor
            width = BASE_WIDTH * (1.0 + 0.1 * speed) * scale_factor
            temp = np.zeros_like(object_grid, dtype=np.uint8)
            processing_t1 = time.time()
            draw_rotated_rectangle(temp, (x, y), (length, width), yaw_deg, value=255)
            timesDraw.append(time.time() - processing_t1)
            processing_t2 = time.time()
            blurred = cv2.GaussianBlur(temp.astype(np.float32), blur_kernel, sigmaX=SIGMA)
            timesGauss.append(time.time() - processing_t2)
            object_grid += blurred / 255.0 * cost
            timesStep.append(time.time() - processing_tstep)
        # Normalize the object grid to ensure max value is 1.0
        object_grid = normalize_grid(object_grid)

        # Merge this object grid into the global grid
        # grid += object_grid
        # Merge this object grid into the global grid and clip the values to avoid oversaturation
        grid = np.clip(grid + object_grid, 0, 1)

    # Normalize the final grid so it doesn't exceed 1.0
    # grid = np.clip(grid / grid.max(), 0, 1)
    print(f"draw time mean: {np.mean(timesDraw)} s")
    print(f"gauss time mean: {np.mean(timesGauss)} s")
    print(f"Step time mean: {np.mean(timesStep)} s")
    print(f"Step time sum: {np.sum(timesStep)} s")
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
