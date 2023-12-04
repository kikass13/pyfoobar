import numpy as np 
import cv2 
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import cdist

from frustrum_culling_gpu import FrustumFilter
from camera_projection_gpu import CameraProjector
from coloredCube import create_colored_cube_array

points = np.load("downsample_ruddington.npy", allow_pickle=True)
xyzrgb = points.astype(np.float32, copy=False)
points_3d = xyzrgb[:,:3].astype(np.float32, copy=False)
colors = (xyzrgb[:,3:].astype(np.float32, copy=False) * 255).astype(np.uint8)
# # Sample every nth point
points_3d = points_3d[::3]
colors = colors[::3]
# print("Downsampled Points: %s" % len(points_3d))

### testing with colored cube
# points_3d, colors = create_colored_cube_array(N=100, size=2.0)
# colors = (colors * 255).astype(np.uint8)

startTime1 = time.time()
frustumFilter = FrustumFilter()
frustumFilter.init()
projector = CameraProjector()
projector.init()

fov = 50.0  # Field of view in degrees
aspect_ratio = 4/3  # Width/height ratio of the viewport
near = 0.2
far = 50.0
focal_length = 1150
image_width = 1200
image_height = 900
fx = focal_length  # Focal length in x-direction   [-fx means a vertical flip of the resulting image]
fy = focal_length  # Focal length in y-direction
cx = image_width / 2.0  # X-coordinate of the principal point
cy = image_height / 2.0  # Y-coordinate of the principal point

# Create the translation vector (in our cooordinate system , before rotation)
# observer_position = np.array([-70.0, 8.0, 1.0], dtype=np.float32)
# observer_position = np.array([50.0, -8.0, 1.0], dtype=np.float32)
observer_position = np.array([-8.0, -70.0, 1.0], dtype=np.float32)
### observer direction into positive x axis
# observer_direction = np.array([1.0, .0, 0]).astype(np.float32)
### observer direction into positive z axis
observer_direction = np.array([0.0, 0.0, 1.0]).astype(np.float32)

##### hmmm default, opencv has to use other convention anyways
#####################
### +x forward, +y up, +z left
# rotation_matrix = np.array([[1, 0, 0],
# 							[0, 1, 0],
# 							[0, 0, 1]], dtype=np.float32) 
# rotation_matrix = np.array([[0, -1, 0],
# 							[0, 0, 1],
# 							[-1, 0, 0]], dtype=np.float32) 
### same as before but left/right switch
rotation_matrix = np.array([[0, 1, 0],	
							[0, 0, 1],
							[-1, 0, 0]], dtype=np.float32) 

# from rotatcheck import generate_rotation_matrices
# for rotation_matrix in generate_rotation_matrices():
if True:
	print(rotation_matrix)
	### switch up z and y? because for some reason our image projection stuff looks differently onto the world
	### in comparison to our frustum filter
	# tvec = np.array([observer_position[0], -observer_position[2], -observer_position[1]])
	tvec = np.array([observer_position[0], -observer_position[2], observer_position[1]])
	#####################
	# Combine rotation and translation into extrinsic matrix
	extrinsic_matrix = np.column_stack((rotation_matrix, tvec))
	extrinsic_matrix = np.vstack([extrinsic_matrix, [0,0,0,1]])
	# Camera matrix (assuming a simple perspective camera)
	camera_matrix = np.array([
		[fx, 0, cx,0],			### -fx means a vertical flip of the resulting image
		[0, fy, cy,0],
		[0, 0, 1, 0]
	], dtype=np.float32)
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
	def filter_points_camera_gpu(points, colors, observer_point, observer_direction, fov, aspect_ratio, near, far):
		frustum_planes = frustumFilter.compute_frustum(observer_point, observer_direction, fov, near, far, aspect_ratio)
		indices = frustumFilter.filter_points_by_frustum_opencl(points, frustum_planes)
		if indices.any():
			return points[indices == 1], colors[indices == 1], frustum_planes
		return np.array([]), np.array([]), np.array([])
	###############################
	begin = time.time()
	### no filter
	filtered_points_3d, filtered_colors = (points_3d, colors)
	### filter dumb
	# filtered_points_3d, filtered_colors = filter_points_behind(points_3d, colors, observer_position, observer_direction)
	### filter ultra smart
	# filtered_points_3d, filtered_colors, frustum_planes = filter_points_camera_gpu(points_3d, colors, observer_position, observer_direction, fov, aspect_ratio, near, far)
	print("FilterDt: %s" % (time.time() - startTime2))
	img = np.zeros((image_height, image_width))
	if filtered_points_3d.any() and filtered_colors.any():
		print(len(filtered_points_3d))
		# projectorMode = "cv2"
		projectorMode = "opencl"
		if projectorMode == "cv2":
		#########################################################
			startTime3 = time.time()
			#############################################################################
			### WHEN USING CV2 PROJECTION FUNCTION
			### cv2 image coordinates
			rotation_matrix = np.array([[0, -1, 0],
										[0, 0, -1],
										[1, 0, 0]], dtype=np.float32)
			# Convert the rotation matrix to a Rodrigues rotation vector
			rvec, _ = cv2.Rodrigues(rotation_matrix)
			### after rotation, the translation for cv2 looks like this
			tvec = np.array([observer_position[1], observer_position[2], -observer_position[0]])
			#############################################################################
			# Project 3D points to 2D image plane
			points_2d, _ = cv2.projectPoints(filtered_points_3d.astype(np.float32), rvec, tvec, camera_matrix, dist_coeffs)
			print("ProjectionDt: %s" % (time.time() - startTime3))
			### Plot 2D points with opencv projected points
			startTimeRender = time.time()
			img = np.zeros((image_height, image_width, 3), dtype=np.uint8) 
			### channels in cv2 is bgr, not rgb - so we switch these up
			# Convert RGB array to BGR array
			for point, color in zip(points_2d.astype(int), filtered_colors):
				try:
					img = cv2.circle(img, tuple(point[0]), 1, color.tolist(), -1) 
				except:
					# print(point)
					pass
			print("RenderDt: %s" % (time.time() - startTimeRender))
		#########################################################
		elif projectorMode == "opencl":
			startTime3 = time.time()
			img = projector.project_points_to_camera_opencl(filtered_points_3d, filtered_colors, extrinsic_matrix, camera_matrix, inflation=6)
			print("ProjectionDt + RenderDt: %s" % (time.time() - startTime3))
		#########################################################

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
		# # Create a mask for black pixels (black would be 0, rather we take really dark pixels <= threshold) 
		mask = (gray <= 30).astype(np.uint8)
		# Separate color channels
		channels = cv2.split(image)
		# Apply inpainting to each channel
		inpainted_channels = [cv2.inpaint(ch, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA) for ch in channels]
		# inpainted_channels = [cv2.inpaint(ch, mask, inpaintRadius=1, flags=cv2.INPAINT_NS) for ch in channels]
		# Merge the inpainted channels
		inpainted_image = cv2.merge(inpainted_channels)
		return inpainted_image
	def simpleImageInterpolate(image):
		###only bottom half of the image
		height, width = image.shape[:2]
		upper_region = image[:height // 2, :]
		lower_region = image[height // 2:, :]
		###
		def find_nearest_non_black(img, row, col, search_radius=10):
			# Define the neighborhood to search for non-black pixels
			neighborhood = img[max(0, row - search_radius):min(img.shape[0], row + search_radius + 1),
				max(0, col - search_radius):min(img.shape[1], col + search_radius + 1)]
			# Get the indices of non-black pixels in the neighborhood
			non_black_indices = np.column_stack(np.where(np.any(neighborhood >= [30, 30, 30], axis=-1)))
			if non_black_indices.size == 0:
				return None  # No non-black pixels in the neighborhood
			# Find the index of the nearest non-black pixel
			distances = np.linalg.norm(non_black_indices - [row, col], axis=1)
			nearest_index = np.argmin(distances)
			# Return the coordinates of the nearest non-black pixel relative to the neighborhood
			return non_black_indices[nearest_index]
		def fill_black_pixels_efficient(img, search_radius=10):
			# Find the coordinates of black pixels
			black_pixels = np.column_stack(np.where(np.all(img <= [30, 30, 30], axis=-1)))
			# Iterate over black pixels and fill with nearest non-black pixel
			for black_pixel in black_pixels:
				nearest_non_black = find_nearest_non_black(img, black_pixel[0], black_pixel[1], search_radius)
				if nearest_non_black is not None:
					img[black_pixel[0], black_pixel[1]] = img[nearest_non_black[0], nearest_non_black[1]]
			return img
		###
		lower_region = fill_black_pixels_efficient(lower_region, search_radius=10)
		filledImage = np.vstack((upper_region, lower_region))
		return filledImage

	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# # plot_frustum(ax, frustum_planes)
	# ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='green', alpha=0.4)
	# plt.title('Scatter Plot of Points with Results')
	# ax.set_xlabel('X-axis')
	# ax.set_ylabel('Y-axis')
	# ax.set_zlabel('Z-axis')
	# plt.legend()
	# ax.set_xlim(-100,100)
	# ax.set_ylim(-20,20)
	# ax.set_zlim(-0.5,10)
	# # Show the plot
	# plt.show()

	startTime4 = time.time()
	# img = cropImage(img, image_width/4.0, image_height/2.5, image_width/2.0, image_height/2.0)
	# img = dilateImage(img)
	# img = resizeImage(img, 2.0)
	# img = interpolateImage(img)
	# img = simpleImageInterpolate(img)
	print("ImageProcDt: %s" % (time.time() - startTime4))
	print("dt: %s" % (time.time() - begin))
	###
	### make image big again for viz
	# img = resizeImage(img, 0.5)
	### channels in cv2 is bgr, not rgb - so we switch these up
	# Convert RGB array to BGR array
	img = img[:, :, ::-1]
	cv2.imshow('Image', img) 
	cv2.waitKey(0) 
	cv2.destroyAllWindows() 
