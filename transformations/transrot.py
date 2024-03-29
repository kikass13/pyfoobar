import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import math

def printP(p):
    print("x: %s, y: %s, z: %s" % (p[0], p[1], p[2]))

def transform(point, trafo):
  p = np.dot(trafo,np.transpose(point))
  return p

x = 1
y = 1
z = 1
p = [x,y,z,1]
printP(p)
### translation
tx = 1.0
ty = 2.0
tz = -0.5
trafo= [
    [1,0,0,tx],
    [0,1,0,ty],
    [0,0,1,tz],
    [0,0,0,1]
] 
newp = transform(p, trafo)
printP(newp)
### rotation via rotation matrix
import quaternionToRotationMat
import eulerAnglesToQuat
deg = 10.0
rad = deg * math.pi / 180.0
Q = eulerAnglesToQuat.get_quaternion_from_euler(0,0,rad)
print("Q: %s" % Q)
R = quaternionToRotationMat.quaternion_rotation_matrix(Q)
print("R:\n%s" % R)
trafo= [
    [R[0][0],R[0][1],R[0][2], 0],
    [R[1][0],R[1][1],R[1][2], 0],
    [R[2][0],R[2][1],R[2][2], 0],
    [0,      0,      0,       1]
]
rotp = transform(newp, trafo)
printP(rotp)

### plot everything
plt.plot(p[0], p[1], color='black', marker='x')
plt.plot(newp[0], newp[1], color='green', marker='o')
plt.plot(rotp[0], rotp[1], color='red', marker='o')
plt.show()
