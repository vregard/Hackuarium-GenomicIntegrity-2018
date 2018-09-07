# Python program for Detection of a
# specific color(blue here) using OpenCV with Python
import cv2
import numpy as np

# Webcamera no 0 is used to capture the frames
#cap = cv2.VideoCapture(0)

# This drives the program into an infinite loop.
#while (1):
    # Captures the live stream frame-by-frame
    #_, frame = cap.read()
    # Converts images from BGR to HSV
frame = cv2.imread('test3.png')
#blursSize = 11
#frame = cv2.GaussianBlur(frame, (blursSize, blursSize), 0)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
xyz = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)



lower_red = np.array([110, 50, 50])
upper_red = np.array([130, 255, 255])

# Here we are defining range of bluecolor in HSV
# This creates a mask of blue coloured
# objects found in the frame.
mask = cv2.inRange(hsv, lower_red, upper_red)

# The bitwise and of the frame and mask is done so
# that only the blue coloured objects are highlighted
# and stored in res
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow('frame', frame)
cv2.imshow('hsv', hsv)

cv2.imwrite('hsv.png', hsv)

cv2.imshow('rgb', rgb)
cv2.imshow('xyz', xyz)
cv2.imshow('YCrCb', YCrCb)

b = hsv.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0
cv2.imshow('B-RGB.jpg',b)

ret, bbis = cv2.threshold(b, 45, 255, cv2.THRESH_BINARY_INV)
#bgrey = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
cv2.imshow('bbis',bbis)

g = hsv.copy()
# set green and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0
cv2.imshow('G-RGB.jpg',g)

ret, gbis = cv2.threshold(g, 90, 255, cv2.THRESH_BINARY_INV)
#bgrey = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
cv2.imshow('gbis',gbis)

r = hsv.copy()
# set green and red channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0
cv2.imshow('R-RGB.jpg',r)

ret, rbis = cv2.threshold(r, 110, 255, cv2.THRESH_BINARY_INV)
#bgrey = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
cv2.imshow('rbis',rbis)

cv2.imshow('mask', mask)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
#k = cv.waitKey(5) & 0xFF
 #   if k == 27:
  #      break
#cv.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('frame',frame)

# Idea 2: Find contour of the nucleus
# Not that good because the threshold for the black and white image changes depending on the image

tmp = cv2.imread('test3.png')
tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

blursSize = 15
tmp = cv2.GaussianBlur(tmp, (blursSize, blursSize), 0)
cv2.imshow('tmp', tmp)

#ret, tmp = cv2.threshold(tmp, 100, 255, cv2.THRESH_BINARY)
ret, tmp = cv2.threshold(tmp, 95, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold', tmp)

#tmp = hsv
# Additional Part with Blobs Detection
#tmp = cv2.bitwise_not(tmp)

# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector_create()

# Detect blobs
#keypoints = detector.detect(tmp)

# Draw detected blobs as red circles.
#  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, np.array([]), (0, 0, 255),
#                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)

#cv2.waitKey(0)

#cv2.imshow('threshold',tmp)
tmp = cv2.bitwise_not(tmp)
#cv2.imshow('not threshold',tmp)

#im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(frame, contours, -1, (0,255,0), 3)
#cv2.drawContours(frame, contours, -1, 255, -1)
cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
cv2.imshow("Contour", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Idea 3: use image minus background instead of the original image alone



cv2.waitKey(0)
cv2.destroyAllWindows()
