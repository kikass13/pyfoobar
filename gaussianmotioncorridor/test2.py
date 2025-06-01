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

def main():
    # Grid size
    H, W = 100, 100
    grid = np.zeros((H, W), dtype=np.float32)

    # Object parameters
    n_objects = 2
    np.random.seed(0)
    positions = np.random.uniform(10, 90, size=(n_objects, 2))
    angles = np.random.uniform(0, 2 * np.pi, size=n_objects)
    speeds = np.random.uniform(5, 15, size=n_objects)

    n_steps = 15
    DT_LOOKAHEAD_FACTOR = 2.0
    DT_FOOTPRINT_SCALE_FACTOR = 2.0
    blur_kernel = (21, 21)
    sigma = 3.0

    BASE_WIDTH = 2.0
    BASE_LENGTH = 4.5

    start_points = []
    vel_vectors = []

    processing_t0 = time.time()
    for i in range(n_objects):
        cx, cy = positions[i]
        vx, vy = pol2cart(speeds[i], angles[i])
        yaw_rad = np.arctan2(vy, vx)
        yaw_deg = np.degrees(yaw_rad)

        start_points.append((cx, cy))
        vel_vectors.append((vx, vy))

        dt = compute_deceleration_dt(vx, vy)
        dt *= DT_LOOKAHEAD_FACTOR

        for step in range(n_steps):
            t = step / (n_steps - 1)
            elapsed_time = t * dt
            x = cx + vx * elapsed_time
            y = cy + vy * elapsed_time

            cost = cost_distribution_based_on_elapsed_time(t, sharpness=0.5, min_cost=0.25)
            
            # Scale rectangle size with speed and t
            scale_factor = 1.0 + (DT_FOOTPRINT_SCALE_FACTOR * t)  # Gradually grows with time
            speed = np.hypot(vx, vy)
            length = BASE_LENGTH * (1.0 + 0.05 * speed) * scale_factor
            width = BASE_WIDTH * (1.0 + 0.1 * speed) * scale_factor

            temp = np.zeros_like(grid, dtype=np.uint8)
            draw_rotated_rectangle(temp, (x, y), (length, width), yaw_deg, value=255)
            blurred = cv2.GaussianBlur(temp.astype(np.float32), blur_kernel, sigmaX=sigma)
            grid += blurred / 255.0 * cost


    # Normalize
    grid = np.clip(grid / grid.max(), 0, 1)
    print(f"processing time: {time.time() - processing_t0:.3f} s")

    # Plotting
    plt.figure(figsize=(7, 7))
    plt.imshow(grid, origin='lower', cmap='hot', extent=[0, W, 0, H])
    plt.colorbar(label="Motion Cost")
    plt.title("Rotated Rectangle Costmap with Velocity Vectors")
    plt.grid(alpha=0.3)

    for (cx, cy), (vx, vy) in zip(start_points, vel_vectors):
        plt.plot(cx, cy, 'go', markersize=8)
        plt.arrow(cx, cy, vx * 0.5, vy * 0.5, head_width=1.5, head_length=2, fc='cyan', ec='cyan')

    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
