import numpy as np 
import cv2 
import time

points = np.load("downsample_ruddington.npy", allow_pickle=True)
xyzrgb = points.astype(np.float32, copy=False)
points_3d = xyzrgb[:,:3].astype(np.float32, copy=False)
colors = xyzrgb[:,3:].astype(np.float32, copy=False) * 255
# Sample every nth point
points_3d = points_3d[::2]
colors = colors[::2]
print("Downsampled Points: %s" % len(points_3d))


startTime1 = time.time()

focal_length = 600
image_width = 1200
image_height = 900

rotation_matrix = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]], dtype=np.float32)

# Convert the rotation matrix to a Rodrigues rotation vector
rvec, _ = cv2.Rodrigues(rotation_matrix)

# Create the translation vector (in our cooordinate system , before rotation)
originaltvec = np.array([-40, -8, 0.5], dtype=np.float32)

### after rotation, the translation looks like this
tvec = np.array([-originaltvec[1], originaltvec[2], -originaltvec[0]])

# Combine rotation and translation into extrinsic matrix
# extrinsic_matrix = np.column_stack((rotation_matrix, tvec))

# Camera matrix (assuming a simple perspective camera)
fx = focal_length  # Focal length in x-direction
fy = focal_length  # Focal length in y-direction
cx = image_width / 2.0  # X-coordinate of the principal point
cy = image_height / 2.0  # Y-coordinate of the principal point

camera_matrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

# Distortion coefficients (assuming no distortion for simplicity)
dist_coeffs = np.zeros((4, 1))
print("InitDt: %s" % (time.time() - startTime1))

startTime2 = time.time()
def filter_points_behind(points, colors, observer_point, observer_direction):
	# Calculate the vector from the observer to the target point
	vector_to_targets = points - observer_point
	# Calculate the dot product
	dot_products = np.dot(vector_to_targets, observer_direction)
	indices_not_behind = np.where(dot_products >= 0)[0]
	points = points[indices_not_behind]
	colors = colors[indices_not_behind]
	return points, colors

### observer direction into positive x axis
# print(len(points_3d))
# observer_point = np.array([-40.0, 0, 0]).astype(np.float32)
observer_direction = np.array([1, 0, 0]).astype(np.float32)
points_3d, colors = filter_points_behind(points_3d, colors, originaltvec, observer_direction)
### channels in cv2 is bgr, not rgb - so we switch these up
# Convert RGB array to BGR array
colors = colors[:, ::-1]

# print(len(points_3d))
print("FilterDt: %s" % (time.time() - startTime2))

startTime3 = time.time()
# Project 3D points to 2D image plane
points_2d, _ = cv2.projectPoints(points_3d.astype(np.float32), rvec, tvec, camera_matrix, dist_coeffs)
print("ProjectionDt: %s" % (time.time() - startTime3))

# Plot 2D points 
img = np.zeros((image_height, image_width, 3), dtype=np.uint8) 
for point, color in zip(points_2d.astype(int), colors):
	try:
		img = cv2.circle(img, tuple(point[0]), 1, color.tolist(), -1) 
	except:
		# print(point)
		pass

def cropImage(image, x, y, width, height):
	image = image[int(y):int(y)+int(height), int(x):int(x)+int(width), :]
	return image
def resizeImage(img, factor):
	h,w = img.shape[:2]
	img = cv2.resize(img, (int(h // factor), int(w // factor)))
	return img
def dilateImage(image):
	# Get the height of the image
	height, width = image.shape[:2]
	# Create a custom kernel
	kernel_size = 3
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	# Define the region for different dilation amounts
	upper_region = image[:height // 2, :]
	lower_region = image[height // 2:, :]
	# Dilation factor for upper and lower regions
	dilation_factor_upper = 1
	dilation_factor_lower = 1
	# Dilate the upper and lower regions with different amounts
	dilated_upper = cv2.dilate(upper_region, kernel, iterations=dilation_factor_upper)
	dilated_lower = cv2.dilate(lower_region, kernel, iterations=dilation_factor_lower)
	# Combine the dilated regions
	dilated_image = np.vstack((dilated_upper, dilated_lower))
	return dilated_image
def interpolateImage(image):
	# # Convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# # Create a mask for black pixels
	mask = (gray == 0).astype(np.uint8)
	# Separate color channels
	channels = cv2.split(image)
	# Apply inpainting to each channel
	inpainted_channels = [cv2.inpaint(ch, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA) for ch in channels]
	# inpainted_channels = [cv2.inpaint(ch, mask, inpaintRadius=1, flags=cv2.INPAINT_NS) for ch in channels]
	# Merge the inpainted channels
	inpainted_image = cv2.merge(inpainted_channels)
	return inpainted_image

startTime4 = time.time()
img = cropImage(img, image_width/4.0, image_height/3.0, image_width/2.0, image_height/2.0)
# img = dilateImage(img)
img = resizeImage(img, 2.0)
img = interpolateImage(img)
print("ImageProcDt: %s" % (time.time() - startTime4))

###
### make image big again for viz
img = resizeImage(img, 0.5)
cv2.imshow('Image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
