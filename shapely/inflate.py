import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.plotting import plot_polygon, plot_line

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
fig = plt.figure(1, figsize=[100,100], dpi=90)

# 1
ax = fig.add_subplot(221)
plot_line(line, ax=ax, add_points=False, color="gray", linewidth=3)
dilated = line.buffer(0.5, cap_style=3)
plot_polygon(dilated, ax=ax, add_points=False, color="blue", alpha=0.5)
ax.set_title('a) dilation, cap_style=3')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)

#2
ax = fig.add_subplot(222)
plot_polygon(dilated, ax=ax, add_points=False, color="gray", alpha=0.5)
eroded = dilated.buffer(-0.3)
plot_polygon(eroded, ax=ax, add_points=False, color="blue", alpha=0.5)
ax.set_title('b) erosion, join_style=1')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)

#3
poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
ax = fig.add_subplot(223)
plot_polygon(poly, ax=ax, add_points=False, color="gray", alpha=0.5)
inflated = poly.buffer(0.5, join_style=2)
plot_polygon(inflated, ax=ax, add_points=False, color="blue", alpha=0.5)
print(list(inflated.exterior.coords))
ax.set_title('c) inflation polygon')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)

#4
line = LineString([(2, 2), (0, 0)])
ax = fig.add_subplot(224)
plot_line(line, ax=ax, add_points=False, color="gray", linewidth=3)
dilated = line.buffer(0.5, cap_style=3)
plot_polygon(dilated, ax=ax, add_points=False, color="blue", alpha=0.5)
print(list(dilated.exterior.coords))
ax.set_title('d) line dilation, cap_style=3')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
plt.show()