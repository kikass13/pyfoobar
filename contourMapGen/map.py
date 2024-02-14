import cv2
import numpy as np
import os
import json
import sys

### robot start position should always be considered as 0,0 origin
pixel_to_meters_resolution_factor = 0.1 ### 1 pixel = 10 cm

class MapFeature:
    def __init__(self) -> None:
        pass
    def convertToM(self, pixelDist):
        return pixelDist * pixel_to_meters_resolution_factor
class SingleFeature(MapFeature):
    def __init__(self, pixelPosition) -> None:
        super().__init__()
        self.pixelPosition = pixelPosition
        self.position = None
    def generate(self):
        self.position = self.convertToM(self.pixelPosition)
        return self
    def correct(self, offset):
        self.position = self.position - offset
        return self
    def dictify(self):
        return {"position": self.position.tolist()}
class MultiFeature(MapFeature):
    def __init__(self, pixelPositions) -> None:
        super().__init__()
        self.pixelPositions = pixelPositions
        self.positions = []
    def generate(self):
        self.positions = [self.convertToM(p) for p in self.pixelPositions]
        return self
    def correct(self, offset):
        self.positions = self.positions - offset
        return self
    def dictify(self):
        return {"positions": self.positions.tolist()}
class Point(SingleFeature):
    def __init__(self, pixelPosition) -> None:
        super().__init__(pixelPosition)
class Path(MultiFeature):
    def __init__(self, pixelPositions) -> None:
        super().__init__(pixelPositions)
class Obstacles(MultiFeature):
    def __init__(self, pixelPositions) -> None:
        super().__init__(pixelPositions)

def createPoint(contours):
    start = None
    if contours:
        ### we only take 1 start
        contour = contours[0]
        ### get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center = np.array([int(x + w/2), int(y+h/2)])
        start = Point(center)
    return start
def createObstacles(contours):
    obstacles = None
    if contours:
        ### merge all obstacle lines int one single object, for easier processing later
        allPixels = []
        for contour in contours:
            pixelsXY = [point[0] for point in contour]
            allPixels.extend(pixelsXY)
        obstacles = Obstacles(allPixels)
    return obstacles
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
    "start": (lower_blue, upper_blue, createPoint),
    "path": (lower_orange_yellow, upper_orange_yellow, createPath),
    "end": (lower_green, upper_green, createPoint),
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
    identifiedDict = {}
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
        identifiedDict[k] = func(contours)
        if debug:
            cv2.imshow('mask %s'%k, mask)
    return identifiedDict, resultImage
def generate(identifiedDict, debug=False):
    resultDict = {}
    for k, v in identifiedDict.items():
        if v:
            resultDict[k] = v.generate()
    if debug:
        print(resultDict["start"].position)
        print(resultDict["end"].position)
        print(resultDict["path"].positions)
        print(len(resultDict["obstacles"].positions))
    return resultDict
def correctMapOrigin(resultDict, debug=False):
    ### correct origin of map objects to be in relation to robot start, where start is 0,0
    corectedDict = {}
    offset = resultDict["start"].position
    for k, v in resultDict.items():
        if v:
            corectedDict[k] = v.correct(offset)
    if debug:
        print(corectedDict["start"].position)
        print(corectedDict["end"].position)
        print(corectedDict["path"].positions)
        print(len(corectedDict["obstacles"].positions))
    return resultDict
def output(outpath, d, debug=False):
    outDict = {}
    for k, v in d.items():
        if v:
            outDict[k] = v.dictify()
    print(outDict)
    with open(outpath, 'w') as file:
        json.dump(outDict, file, indent=4)

#######################################################################################

def main():
    debug = True
    input = sys.argv[1]
    hsvImage = prepare(input, debug=debug)
    identifiedDict, resultImage = process(hsvImage, debug=debug)
    resultDict = generate(identifiedDict, debug=debug)
    correctedDict = correctMapOrigin(resultDict, debug=debug)
    output(os.path.splitext(input)[0] + ".scenario", correctedDict, debug=debug)
    if debug:
        cv2.imshow('ResultImage', resultImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()