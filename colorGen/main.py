import random

# Function to generate a distinct RGB color for an index
def generate_distinct_color(index, total_indices):
    hue = (index / total_indices) * 360  # Spread out hues in the range [0, 360)
    saturation = 1.0  # Keep saturation constant
    value = 1.0  # Keep value (brightness) constant

    # Convert HSV to RGB
    c = saturation * value
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c

    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # Scale RGB values to the [0, 255] range
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

    return r, g, b

# Number of distinct colors needed
total_colors = 1000
colors = [generate_distinct_color(i, total_colors) for i in range(total_colors)]

# Randomly shuffle the list in place
# random.shuffle(colors)

# Print the generated colors
for i, color in enumerate(colors):
    r, g, b = color
    print(f"Index {i}: R={r}, G={g}, B={b}")
