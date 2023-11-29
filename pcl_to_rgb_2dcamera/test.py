import numpy as np 
import cv2 

# Define intrinsic camera parameters 
focal_length = 500
image_width = 640
image_height = 480
intrinsic_matrix = np.array([ 
	[focal_length, 0, image_width/2], 
	[0, focal_length, image_height/2], 
	[0, 0, 1] 
]) 

# Define extrinsic camera parameters 
rvec = np.array([0, 0, 0], dtype=np.float32) 
tvec = np.array([0, 0, 100], dtype=np.float32) 

# Generate 3D points on a paraboloid 
u_range = np.linspace(-1, 1, num=20) 
v_range = np.linspace(-1, 1, num=20) 
u, v = np.meshgrid(u_range, v_range) 
x = u 
y = v 
z = u**2 + v**2

points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3) 
print(points_3d)
random_colors = np.random.randint(0, 256, size=(len(points_3d), 3), dtype=np.uint8)

# Project 3D points onto 2D plane 
points_2d, _ = cv2.projectPoints(points_3d, 
								rvec, tvec.reshape(-1, 1), 
								intrinsic_matrix, 
								None) 

# Plot 2D points 
img = np.zeros((image_height, image_width, 3), dtype=np.uint8) 

for point,color in zip(points_2d.astype(int), random_colors): 
	img = cv2.circle(img, tuple(point[0]), 1, color.tolist(), -1) 

cv2.imshow('Image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
