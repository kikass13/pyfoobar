
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

import quaternionToRotationMat
import eulerAnglesToQuat

class Point():
    def __init__(self, x,y, z = 0.0):
        self.x = x
        self.y = y
        self.z = z
    def np(self):
        return np.array((self.x, self.y, self.z))
    def __repr__(self) -> str:
        return "[%s, %s, %s]" % (self.x, self.y, self.z)
def transform(point:Point, trafo):
    v = np.append(point.np(), 1)
    p = np.dot(trafo,np.transpose(v))
    return Point(p[0], p[1], z=p[2])

def translation_vector(x,y,z):
    return np.array([x,y,z])
def transformation_matrix(T, R):
    return [
        [R[0][0],R[0][1],R[0][2], T[0]],
        [R[1][0],R[1][1],R[1][2], T[1]],
        [R[2][0],R[2][1],R[2][2], T[2]],
        [0,      0,      0,       1]
    ]


length = 2.0
width = 1.0
rect = [
    Point(0.0,width/2.0), 
    Point(0, -width/2.0),
    Point(length, -width/2.0),
    Point(length, width/2.0), 
    Point(0.0,width/2.0), 
]
x = 5.0
y = 3.0
z = 0.0

deg = 45.0
rad = deg * math.pi / 180.0
Q = eulerAnglesToQuat.get_quaternion_from_euler(0,0,rad)
R = quaternionToRotationMat.quaternion_rotation_matrix(Q)
T = translation_vector(x,y,z)
trafo = transformation_matrix(T,R)

print(trafo)
rotRect = []
for p in rect:
    rotRect.append(transform(p, trafo))

### plot everything
xL = [p.x for p in rotRect]
yL = [p.y for p in rotRect]
plt.plot(xL, yL, "go-", color='black')
xL = [p.x for p in rect]
yL = [p.y for p in rect]
plt.plot(xL, yL, "go-", color='black')
plt.show()
