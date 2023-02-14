import cv2
import numpy as np


np.random.seed(seed=0)
N = 100
X1 = np.random.rand(100,2)
X2 = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
linex = np.linspace(0.0, 5.0, num=N)
liney = np.zeros(N)
X3 = []
for x,y in zip(linex,liney):
    X3.append([x,y])
X3 = np.array(X3)
linex = np.linspace(-1.0, 3.0, num=20)
liney = np.empty(10)
liney.fill(5.0)
X4 = []
for x,y in zip(linex,liney):
    X4.append([x,y])
X4 = np.array(X4)
linex = np.linspace(2.0, 3.0, num=20)
liney = np.empty(100)
liney.fill(6.0)
X5 = []
for x,y in zip(linex,liney):
    X5.append([x,y])
X5 = np.array(X5)
######
# X = X3
X = np.append(X1, X3, 0)
X = np.append(X, X4, 0)
X = np.append(X, X5, 0)
######


# Make an image from discrete points
w = 800
h = 800
scale = 10.0
offsetx = w / 2.0
offsety = h / 2.0
img = np.zeros((h, w), np.uint8)
for i,p in enumerate(X):
    x = int(p[0] * scale + offsetx)
    y = int(p[1] * scale + offsety)
    img[y,x] = 255
gray = cv2.cvtColor(img, cv2.CV_8UC1)
edges = cv2.Canny(gray,h,w,apertureSize = 5)
# cv2.imshow('image',gray)
# cv2.waitKey(0)
# cv2.imshow('image',edges)
# cv2.waitKey(0)
###

result = np.zeros((h, w), np.uint8)
# lines = cv2.HoughLines(edges,1,np.pi/180,20)
# for line in lines:
#     rho = line[0][0]
#     theta = line[0][1]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(result,(x1,y1),(x2,y2),255,2)
minLineLength = 1
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(result,(x1,y1),(x2,y2),255,2)

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.waitKey(0)