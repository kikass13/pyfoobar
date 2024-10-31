import freetype
import numpy as np
import matplotlib.pyplot as plt

# Path to a TrueType font file
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Change the path as needed

# Create a face object from the font file
face = freetype.Face(font_path)
face.set_char_size(48 * 64)  # Set the size of the font in 1/64th points

# Text to render
text = "Hello, World!"

# Store bitmaps and dimensions
bitmaps = []
widths = []
heights = []

# Render each character
for char in text:
    face.load_char(char)
    bitmap = face.glyph.bitmap
    width, height = bitmap.width, bitmap.rows
    bitmap_data = np.zeros((height, width), dtype=np.uint8)

    # Fill the numpy array with the bitmap data
    for y in range(height):
        for x in range(width):
            bitmap_data[y, x] = bitmap.buffer[y * bitmap.width + x]

    bitmaps.append(bitmap_data)
    widths.append(width)
    heights.append(height)

# Calculate the total width and max height for the full bitmap
total_width = sum(widths)
max_height = max(heights)

# Create a full bitmap to hold the combined characters
combined_bitmap = np.zeros((max_height, total_width), dtype=np.uint8)

# Combine character bitmaps into the full bitmap
x_offset = 0
for i, bitmap in enumerate(bitmaps):
    height = heights[i]
    width = widths[i]

    # Place the bitmap in the combined bitmap
    combined_bitmap[max_height - height:max_height, x_offset:x_offset + width] = bitmap
    x_offset += width  # Move the offset for the next character

# Create a plot to display the combined bitmap
plt.figure(figsize=(10, 4))
plt.imshow(combined_bitmap, cmap='gray', origin='upper')
plt.title("Combined Bitmap of 'Hello, World!'")
plt.axis('off')  # Turn off the axis
plt.show()
