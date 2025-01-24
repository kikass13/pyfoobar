import numpy as np
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Example points (you can replace this with your own)
points = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 3), (3, 1), (4, 5), (6, 4)]

# Create a concave hull using alphashape
# The alpha parameter controls the "tightness" of the hull
alpha = 0.0  # You can adjust this value to make the hull more or less tight
concave_hull = alphashape.alphashape(points, alpha)
print(concave_hull)

# Visualize the concave hull
plt.figure(figsize=(8, 8))

# Plot the points
x_points, y_points = zip(*points)
plt.scatter(x_points, y_points, color='red', label='Points')

# Plot the concave hull
if isinstance(concave_hull, Polygon):
    x_hull, y_hull = concave_hull.exterior.xy
    plt.fill(x_hull, y_hull, color='lightblue', alpha=0.5, label='Concave Hull')
    plt.plot(x_hull, y_hull, color='blue', linewidth=2)

# Finalize plot
plt.title("Concave Hull (Elastic Band Effect)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()