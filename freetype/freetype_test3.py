import freetype
import numpy as np
import time
import os
import sys
import glob
import matplotlib.pyplot as plt

def getAllAvailableFonts(dir = "/usr/share/fonts/truetype"):
    fontPaths = glob.glob(os.path.join(dir, '**', '*.ttf'), recursive=True)
    fonts = [os.path.basename(path).split(".")[0] for path in fontPaths]
    return {font:path for font, path in zip(fonts, fontPaths)}
AvailableFonts = getAllAvailableFonts()

FONT_SIZE_Y = 64
def renderTextOutlines(text, font, font_size=48, connectEnds=True):
    fontPath = AvailableFonts[font]
    face = freetype.Face(fontPath)
    # face.set_char_size(char_size_x * char_size_y)
    face.set_char_size(font_size * FONT_SIZE_Y, 0)  # Set the size and resolution 
    outlines = []
    x_offset = 0
    for char in text:
        face.load_char(char)
        outline : freetype.Outline = face.glyph.outline
        contourPoints = []
        index = 0
        for endPointIndex in outline.contours:
            relevantPoints = np.array(outline.points[index:endPointIndex+1])
            if relevantPoints.any():
                relevantPoints = relevantPoints / 64 + np.array([x_offset, 0])
                ### add start point again so that the las line connects to the start
                if connectEnds:
                    relevantPoints = np.vstack([relevantPoints, relevantPoints[0]])
                contourPoints.append(relevantPoints)
            index = endPointIndex+1
        outlines.append(contourPoints)
        x_offset += face.glyph.advance.x / 64
    return outlines


if __name__ == '__main__':
    # print(AvailableFonts)
    font = "DejaVuSans"
    # font = "Ubuntu-B"
    start = time.time()
    ### draw each character's outline
    outlines = renderTextOutlines("abcdeozm1234567890!", font, font_size=8)
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    for contourPoints in outlines:
        ### go through each contour point list
        for i, points in enumerate(contourPoints):
            if points.any():
                ax.plot(points[:, 0], points[:, 1], linewidth=2, color="black")
                ax.fill(*zip(*points), color='red')
    ax.set_aspect('equal')
    ax.set_title("Vector Graphics of 'Hello, World!'")
    # plt.axis('off')  # Turn off the axis
    plt.show()
