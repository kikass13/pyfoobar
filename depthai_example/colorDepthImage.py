
import numpy as np
import skimage.exposure
import cv2

def color_depth_image(img):
    # min_depth = np.min(img)
    # max_depth = np.max(img)
    # normalized_depth_image = (img - min_depth) / (max_depth - min_depth)     
    # Normalize depth values to range [0, 1]
    image_float = img.astype(np.float32)
    normalized_depth_image = cv2.normalize(image_float, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Apply colormap (Jet colormap is commonly used for depth images)
    artificialScalingFactor = 8.0
    colored_depth_image = cv2.applyColorMap(np.uint8(normalized_depth_image * artificialScalingFactor * 255), cv2.COLORMAP_JET)
    return colored_depth_image
