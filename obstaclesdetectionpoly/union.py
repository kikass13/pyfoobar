from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Example rectangles (you can replace with your actual data)
rectangle1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
rectangle2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])  # Adjacent to rectangle1
rectangle3 = Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])  # Separate from others

# Combine the rectangles (tight union)
multi_rectangles = MultiPolygon([rectangle1, rectangle2, rectangle3])
combined = unary_union(multi_rectangles)

# Visualize the result
plt.figure(figsize=(8, 8))

# Plot each rectangle in the MultiPolygon
for poly in multi_rectangles.geoms:
    x, y = poly.exterior.xy
    plt.fill(x, y, color='lightblue', alpha=0.5, label='Rectangles' if poly == multi_rectangles.geoms[0] else None)
    plt.plot(x, y, color='blue')

# Plot the combined result
if isinstance(combined, Polygon):
    x_combined, y_combined = combined.exterior.xy
    plt.fill(x_combined, y_combined, color='lightgreen', alpha=0.5, label='Combined')
    plt.plot(x_combined, y_combined, color='green', linewidth=2)

# Finalize plot
plt.title("Union of Rectangles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
