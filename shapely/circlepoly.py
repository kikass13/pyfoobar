from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line


radius = 5
center = Point(float(0), float(0))
poly = center.buffer(radius, cap_style=3)

for x,y in poly.exterior.coords:
    print(x,y)

fig, ax = plt.subplots()
plot_polygon(poly, ax=ax, add_points=True, color="blue", alpha=0.5)
ax.axis("equal")
ax.grid(True)
plt.show()
