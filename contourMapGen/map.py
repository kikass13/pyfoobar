import cv2
import numpy as np

### robot start position should always be considered as 0,0 origin
pixel_to_meters_resolution_factor = 0.1 ### 1 pixel = 10 cm

class MapFeature:
    def __init__(self, pixels) -> None:
        self.pixels = pixels
    def convertToM(self, pixelDist):
        return pixelDist * pixel_to_meters_resolution_factor
class Start(MapFeature):
    def __init__(self, pixels) -> None:
        super().__init__(pixels)
class End(MapFeature):
    def __init__(self, pixels) -> None:
        super().__init__(pixels)
class Obstacle(MapFeature):
    def __init__(self, pixels) -> None:
        super().__init__(pixels)
class Path(MapFeature):
    def __init__(self, pixels) -> None:
        super().__init__(pixels)



def createStart(contours):
    pass
def createEnd(contours):
    pass
def createObstacles(contours):
    pass
def createPath(contours):
    path = None
    if contours:
        ### we only take 1 path
        contour = contours[0]
        pixelsXY = [point[0] for point in contour]
        path = Path(pixelsXY)
    return path

lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_orange_yellow = np.array([5, 100, 100])
upper_orange_yellow = np.array([30, 255, 255])
lower_green = np.array([40, 40, 40]) 
upper_green = np.array([80, 255, 255])
lower_black = np.array([0, 0, 0])  
upper_black = np.array([179, 255, 30])

map_dict = {
    "start": (lower_blue, upper_blue, createStart),
    "path": (lower_orange_yellow, upper_orange_yellow, createPath),
    "end": (lower_green, upper_green, createEnd),
    "obstacles": (lower_black, upper_black, createObstacles),
}

#######################################################################################

def prepare(imgPath, debug=False):
    ### load image
    image = cv2.imread(imgPath)
    ### convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if debug:
        cv2.imshow('Original', image)
    return hsv
def process(image, debug=False):
    ### image should be hsv format
    resultDict = {}
    resultImage = np.zeros_like(image)
    for k, content in map_dict.items():
        lower,upper, func = content
        ### threshold the HSV image to get color range
        mask = cv2.inRange(image, lower, upper)
        ### find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug:
            ### show all contours found in green , debugs
            cv2.drawContours(resultImage, contours, -1, (0, 255, 0), 2)
        ### call generator func and save result
        resultDict[k] = func(contours)
        if debug:
            cv2.imshow('mask %s'%k, mask)
    return resultDict, resultImage
def correctMapOrigin(resultDict, debug=False):
    ### correct origin of map objects to be in relation to robot start, where start is 0,0
    print(resultDict["path"].pixels)
    return resultDict

#######################################################################################

def main():
    debug = True
    hsvImage = prepare("map_image.bmp", debug=debug)
    resultDict, resultImage = process(hsvImage, debug=debug)
    correctedDict = correctMapOrigin(resultDict, debug=debug)
    if debug:
        cv2.imshow('ResultImage', resultImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()