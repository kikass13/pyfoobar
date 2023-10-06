import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d
import time


def colorBetween(minV, maxV, val):
    return max(0, min(1, (val - minV) / (maxV-minV)))

def createCells(grid, top, right, rows, cols, gridsize):
    rects = []
    minZ = -3
    maxZ = 3
    for row in range(rows):
        y = top - row * gridsize
        for col in range(cols):
            x = right + col * gridsize
            r = 1
            g = 1
            b = 1
            if not np.isnan(grid[row,col]):
                r = colorBetween(minZ, maxZ, grid[row,col])
                g = 0
                b = 0
            rectangle = plt.Rectangle((x, y), gridsize, gridsize, edgecolor='black', facecolor=(r,g,b), linewidth=1)
            rects.append(rectangle)
    return rects

def pointsToGrid(points, gridsize=0.25, length=(0, 100.0), width=(-20.0, 20.0)):
    xn, xp = length
    yn, yp = width
    rows = int((xp - xn) / gridsize)
    cols = int((yp - yn) / gridsize)
    grid = np.full((rows, cols), np.nan)
    print("=============== r:%s c:%s" % (rows, cols))
    def getIndex(x,y):
        xi = int(np.fabs(x - xn) / gridsize)
        yi = int(np.fabs(y - yn) / gridsize)
        return xi,yi 
    for (x,y,z) in points:
        if x > xn and x < xp and y > yn and y < yp:
            xi,yi = getIndex(x,y)
            if not np.isnan(grid[xi,yi]):
                grid[xi,yi] = max(grid[xi,yi], z)
            else:
                grid[xi,yi] = z

    ### create drawable cells
    cells = createCells(grid, xp, yn, rows,cols, gridsize)
    return grid, cells, (xn, xp), (yn, yp)

gridsize = 1.0

# Read point cloud:
pcd = o3d.io.read_point_cloud("nonground.pcd")
# Get points and transform it to a numpy array:
points = np.asarray(pcd.points)

print(points)

start = time.time()

grid, cells, length, width = pointsToGrid(points, gridsize=gridsize)

figure, ax = plt.subplots(1)

print(time.time() - start)

print(grid[:])


figure, ax = plt.subplots(1)
plt.xlim(-20,20)
plt.ylim(0, 100)
for cell in cells:
    ax.add_patch(cell)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()


# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]
# # grid size or so to say average distance between grid cells
# xi = np.arange(length[0], length[1], gridsize)
# yi = np.arange(width[0], width[1], gridsize)
# zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')
# Visualization of the result
# f = plt.figure(figsize=(12,6))
# CS = plt.contour(xi,yi,grid[:],15,linewidths=0.11,colors='k')
# CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
# plt.colorbar(CS)
# plt.show()

