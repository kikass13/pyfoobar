import numpy as np
import matplotlib.pyplot as plt
import cv2

# Grid parameters
H, W = 100, 100
grid = np.zeros((H, W), dtype=np.float32)

# Object motion parameters
n_objects = 2
np.random.seed(0)
positions = np.random.uniform(10, 90, size=(n_objects, 2))
angles = np.random.uniform(0, 2*np.pi, size=n_objects)
speeds = np.random.uniform(5, 10, size=n_objects)
dt = 1.0

# Gaussian kernel size (how much "spread" per object)
blur_kernel = (11, 11)  # Must be odd
sigma = 3.0             # Higher = more blur (uncertainty)

# Draw each object's motion line into the grid
for i in range(n_objects):
    cx, cy = positions[i]
    vx = speeds[i] * np.cos(angles[i])
    vy = speeds[i] * np.sin(angles[i])

    # Current and future positions
    x0, y0 = int(cx), int(cy)
    x1, y1 = int(cx + vx * dt * 2), int(cy + vy * dt * 2)

    # Create a temporary image to draw the line
    temp = np.zeros_like(grid, dtype=np.uint8)
    cv2.line(temp, (x0, y0), (x1, y1), color=255, thickness=2)
    
    # Blur the line to create a "corridor"
    blurred = cv2.GaussianBlur(temp.astype(np.float32), blur_kernel, sigma)
    
    # Add to main grid
    grid += blurred / 255.0  # Normalize since temp is [0–255]

# Normalize final grid
grid = np.clip(grid / grid.max(), 0, 1)

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(grid, origin='lower', cmap='hot', extent=[0, W, 0, H])
plt.colorbar(label="Occupancy Intensity")
plt.title("Gaussian Motion Corridors (Fast Approximation)")
plt.grid(alpha=0.3)
plt.show()
