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

# Create a plot to display the bitmap for each character
plt.figure(figsize=(10, 4))

# Set initial position for plotting
x_offset = 0

for i, bitmap in enumerate(bitmaps):
    height = heights[i]
    width = widths[i]

    # Display each character bitmap
    plt.subplot(1, len(bitmaps), i + 1)
    plt.imshow(bitmap, cmap='gray', origin='upper')
    plt.title(f"'{text[i]}'")
    plt.axis('off')  # Turn off the axis

    # Adjust the x-axis limits to space out the characters
    plt.xlim(-1, width + 1)
    plt.ylim(-1, height + 1)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
