
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)
N = 100
X1 = np.random.rand(100,2)
X2 = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
linex = np.linspace(0.0, 5.0, num=N)
liney = np.zeros(N)
X3 = []
for x,y in zip(linex,liney):
    X3.append([x,y])
X3 = np.array(X3)
linex = np.linspace(2.0, 3.0, num=10)
liney = np.empty(10)
liney.fill(5.0)
X4 = []
for x,y in zip(linex,liney):
    X4.append([x,y])
X4 = np.array(X4)
linex = np.linspace(2.0, 3.0, num=10)
liney = np.empty(10)
liney.fill(6.0)
X5 = []
for x,y in zip(linex,liney):
    X5.append([x,y])
X5 = np.array(X5)
######
X = np.append(X1, X3, 0)
X = np.append(X, X4, 0)
X = np.append(X, X5, 0)
######
kmeans = KMeans(random_state=0, n_init="auto").fit(X)
kmeans.predict(X)
result = kmeans.cluster_centers_

x = X[:,0]
y = X[:,1]
rx = result[:,0]
ry = result[:,1]

plt.plot(x, y, "bo")
plt.plot(rx, ry, "ro")
plt.axis("equal")
plt.grid(True)
plt.show()