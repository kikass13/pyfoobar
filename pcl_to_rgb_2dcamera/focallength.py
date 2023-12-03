import math

def fov_to_focal_length(sensor_dimension, fov_degrees):
    fov_radians = math.radians(fov_degrees)
    return sensor_dimension / (2 * math.tan(fov_radians / 2))

def focal_length_mm_to_pixels(focal_length_mm, pixel_size_um):
    pixel_size_mm = pixel_size_um / 1000  # Convert pixel size to millimeters
    focal_length_pixels = focal_length_mm / pixel_size_mm
    return focal_length_pixels

# Example usage
sensor_dimension = 20  # Full-frame sensor dimension in millimeters
fov_degrees = 60       # Desired FOV in degrees

# Calculate focal length
focal_length_mm = fov_to_focal_length(sensor_dimension, fov_degrees)

print("Focal Length:", focal_length_mm, "mm")

pixel_size_um = 15     # Pixel size in micrometers
# Convert focal length to pixels
focal_length_pixels = focal_length_mm_to_pixels(focal_length_mm, pixel_size_um)

print("Focal Length in Pixels:", focal_length_pixels)