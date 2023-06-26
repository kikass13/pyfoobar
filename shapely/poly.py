from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line



xy = [
        (2.349090052900678, 3.1412292082060707),
        (2.349090052900678, 2.784415029383825),
        (1.9922758740784323, 2.784415029383825),
        (1.9922758740784323, 3.1412292082060707),
        (2.349090052900678, 3.1412292082060707),
]
poly = Polygon(xy)

fig, ax = plt.subplots()
plot_polygon(poly, ax=ax, add_points=True, color="blue", alpha=0.5)
ax.axis("equal")
ax.grid(True)
plt.show()
