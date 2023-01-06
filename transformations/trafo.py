import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

def linear_transformation(src, a):
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    new_points = np.linalg.inv(a).dot(points).round().astype(int)
    x, y = new_points.reshape((2, M, N), order='F')
    indices = x + N*y
    return np.take(src, indices, mode='wrap')

mpl.rcParams.update({'image.cmap': 'Accent',
                     'image.interpolation': 'none',
                     'xtick.major.width': 0,
                     'xtick.labelsize': 0,
                     'ytick.major.width': 0,
                     'ytick.labelsize': 0,
                     'axes.linewidth': 0})

###original
aux = np.ones((100, 100), dtype=int)
src = np.vstack([np.c_[aux, 2*aux], np.c_[3*aux, 4*aux]])
plt.imshow(src)
plt.show()
# Scaling the plane in the x-axis by a factor of 1.5:
a = np.array([[1.5, 0],
              [0, 1]])
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()
### Dilating the plane by a factor of 1.8:
a = 1.8*np.eye(2)
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()
### Dilating the plane by a factor of 0.5:
a = .5*np.eye(2)
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()
### Scaling the plane in the y-axis by a factor of 0.5:
a = np.array([[1, 0],
              [0, .5]])
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()
###Shearing about the y-axis with a vertical displacement of +x/2
a = np.array([[1, 0],
              [.5, 1]])
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()
### Rotation through 45 deg about the origin
alpha = np.pi/4
a = np.array([[np.cos(alpha), -np.sin(alpha)],
              [np.sin(alpha), np.cos(alpha)]])
dst = linear_transformation(src, a)
plt.imshow(dst)
plt.show()