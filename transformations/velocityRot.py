
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

import quaternionToRotationMat
import eulerAnglesToQuat
import quaternionMultiply

def angle_dot(a, b):
    dot_product = np.dot(a, b)
    prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
    angle = np.arccos(dot_product / prod_of_norms)
    return dot_product, angle

class Vector3:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self) -> str:
        return "[%s, %s, %s]" % (self.x, self.y, self.z)
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    def np(self):
        return np.array([self.x, self.y, self.z])
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
    
def translation_vector(x,y,z):
    return np.array([x,y,z])


# rotate vector v1 by quaternion q1 
def qv_mult(q1, v):
    q2 = [v.x, v.y, v.z, 0.0]
    q_conjugate = quaternionMultiply.quaternion_conjugate(q1)
    q_mul = quaternionMultiply.quaternion_multiply(q1, q2)
    return quaternionMultiply.quaternion_multiply(q_mul, q_conjugate)[:3]
#twist_rot is the velocity in euler angles (rate of roll, rate of pitch, rate of yaw; angular.x, angular.y, angular.z)
#twist_vel is the velocity in cartesian coordinates (linear.x, linear.y, linear.z)
def transform_velocity(v : Vector3, trafo:Frame):
    q2 = [v.x, v.y, v.z, 0.0]
    q1 = trafo.getQuaternion()
    q_conjugate = quaternionMultiply.quaternion_conjugate(q1)
    q_mul = quaternionMultiply.quaternion_multiply(q1, q2)
    res = quaternionMultiply.quaternion_multiply(q_mul, q_conjugate)[:3]
    return Vector3(res[0], res[1], res[2])

##p1
x1 = 4.0
y1 = 2.0
z1 = 0.0
rol1 = 0.0
pit1 = 0.0
yaw1 = 0.0
v1 = Vector3(3.0, 1.0, 0.0)
##p2 - other pose for frame 
x2 = 2.0
y2 = 2.0
z2 = 0.0
rol2 = 0.0
pit2 = 0.0
yaw2 = 0.0
v2 = Vector3(1.0, 0.0, 0)
########################################
Q = eulerAnglesToQuat.get_quaternion_from_euler(rol2,pit2,yaw2)
T = translation_vector(x2,y2,z2)
trafo = Frame(Q, T)
########################################
v_transformed = transform_velocity(v1, trafo)
v_diff = v_transformed - v2 
v_diff_r =  np.linalg.norm([v_diff.x, v_diff.y])
### calculate v_diff_r for only when the overall direction is the same as ours
### dot product of the two angles
direction, phi = angle_dot(v_transformed.np(), v2.np())
    # v_diff_r_relative = 
print(direction, phi)
########################################
### plot v1
plt.plot(x1, y1, ".", color='red')
arrow_length = np.linalg.norm([v1.x, v1.y])
arrow_angle = math.atan2(v1.y, v1.x)
arrow_end_x = x1+math.cos(arrow_angle)*arrow_length
arrow_end_y = y1+math.sin(arrow_angle)*arrow_length
dx = arrow_end_x-x1
dy = arrow_end_y-y1
plt.arrow(x1, y1, dx, dy, color='red')
print("v1 = %s" % arrow_length)
##plot v2
plt.plot(x2, y2, ".", color='red')
arrow_length = np.linalg.norm([v2.x, v2.y])
arrow_angle = math.atan2(v2.y, v2.x)
arrow_end_x = x2+math.cos(arrow_angle)*arrow_length
arrow_end_y = y2+math.sin(arrow_angle)*arrow_length
dx = arrow_end_x-x2
dy = arrow_end_y-y2
plt.arrow(x2, y2, dx, dy, color='orange')
print("v2 = %s" % arrow_length)
# ##plot vdiff
arrow_length = v_diff_r
arrow_angle = phi
arrow_end_x = math.cos(arrow_angle)*arrow_length
arrow_end_y = math.sin(arrow_angle)*arrow_length
plt.arrow(0, 0, arrow_end_x, arrow_end_y, color='blue')

print("v_transformed: %s" % v_transformed)
print("v_r: %s" % np.linalg.norm([v_transformed.x, v_transformed.y]))
print("v_diff: %s" % v_diff)
print("v_diff_r: %s" % v_diff_r)
# print("v_twist_diff: %s" % (v_twist_diff))
# ###
ax = plt.gca()
ax.set_xlim([-2, 6])
ax.set_ylim([-2, 6])
plt.show()


