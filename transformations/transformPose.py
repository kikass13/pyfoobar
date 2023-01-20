
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

import quaternionToRotationMat
import eulerAnglesToQuat
import quaternionMultiply

class Frame:
    def __init__(self,Q, T):
        self.q = Q
        self.t = T
    def transformation_matrix(self):
        R = quaternionToRotationMat.quaternion_rotation_matrix(self.q)
        return np.array([
            [R[0][0],R[0][1],R[0][2], self.t[0]],
            [R[1][0],R[1][1],R[1][2], self.t[1]],
            [R[2][0],R[2][1],R[2][2], self.t[2]],
            [0.0,    0.0,    0.0,     1.0]
        ])
    def getQuaternion(self):
        return self.q

class Pose:
    def __init__(self,x,y,z,rol,pit,yaw):
        self.x = x
        self.y = y
        self.z = z
        self.rol = rol
        self.pit = pit
        self.yaw = yaw
    def __repr__(self) -> str:
        return "[%s, %s, %s | %s, %s, %s]" % (self.x, self.y, self.z, self.rol, self.pit, self.yaw)
    
def translation_vector(x,y,z):
    return np.array([x,y,z])

def doTransform(pose : Pose, trafo: Frame):
    ### translation
    m = trafo.transformation_matrix()
    v = np.array((pose.x, pose.y, pose.z, 1.0))
    t = np.dot(m, np.transpose(v))
    ### orientation
    q1 = eulerAnglesToQuat.get_quaternion_from_euler(pose.rol, pose.pit, pose.yaw)
    q2 = trafo.getQuaternion()
    q =  quaternionMultiply.quaternion_multiply(q2, q1)
    r,p,y = eulerAnglesToQuat.euler_from_quaternion(q)
    return Pose(t[0], t[1], t[2], r, p, y)

##p1
x1 = 2.0
y1 = 4.0
z1 = 0.0
rol1 = 0.0
pit1 = 0.0
yaw1 = math.pi/2.0
p1 = Pose(x1,y1,z1,rol1,pit1,yaw1)
##p2 - other pose for frame 
x2 = -1.0
y2 = -1.0
z2 = 0.0
rol2 = 0.0
pit2 = 0.0
yaw2 = math.pi/4.0
Q = eulerAnglesToQuat.get_quaternion_from_euler(rol2,pit2,yaw2)
T = translation_vector(x2,y2,z2)
trafo = Frame(Q, T)
### p3, which shows p1 -> p2
p3 = doTransform(p1, trafo)
print(p3)
########################################
### plot p1
plt.plot(x1, y1, ".", color='red')
arrow_length = 1.0
arrow_end_x = x1+math.cos(yaw1)*arrow_length
arrow_end_y = y1+math.sin(yaw1)*arrow_length
dx = arrow_end_x-x1
dy = arrow_end_y-y1
plt.arrow(x1, y1, dx, dy, color='red')
##plot p2
plt.plot(x2, y2, ".", color='black')
arrow_length = 1.0
arrow_end_x = x2+math.cos(yaw2)*arrow_length
arrow_end_y = y2+math.sin(yaw2)*arrow_length
dx = arrow_end_x-x2
dy = arrow_end_y-y2
plt.arrow(x2, y2, dx, dy, color='black')
##plot p3
plt.plot(p3.x, p3.y, ".", color='green')
arrow_length = 1.0
arrow_end_x = p3.x+math.cos(p3.yaw)*arrow_length
arrow_end_y = p3.y+math.sin(p3.yaw)*arrow_length
dx = arrow_end_x-p3.x
dy = arrow_end_y-p3.y
plt.arrow(p3.x, p3.y, dx, dy, color='green')
###
ax = plt.gca()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
plt.show()



# print(trafo)
# rotRect = []
# for p in rect:
#     rotRect.append(transform(p, trafo))
