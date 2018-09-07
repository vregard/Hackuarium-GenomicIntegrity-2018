def sayhello(who):
    print 'Hello,', who + '!'
    print 'What a lovely day.'


# Standard imports
import numpy as np
import cv2

# Read image
#im = cv2.imread("Screen Shot 2018-07-26 at 16.27.02.png", cv2.IMREAD_GRAYSCALE) # 2 cells
#im = cv2.imread("Screen Shot 2018-07-26 at 16.28.09.png", cv2.IMREAD_GRAYSCALE)
#im = cv2.imread("WhatsApp Image 2018-07-30 at 14.30.07.jpeg", cv2.IMREAD_GRAYSCALE)
im = cv2.imread("Vio4.PNG", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", im)
cv2.waitKey(0)

# Conversion to binary image
# global thresholding
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
# => good definition of the nucleus
# adaptative mean thresholding
th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
# adaptative gaussian thresholding
th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,2)
# => best definition of the contours

cv2.imshow("global thresholding", thresh1)
cv2.waitKey(0)

cv2.imshow("adaptative mean thresholding", th2)
cv2.waitKey(0)

cv2.imshow("adaptative gaussian thresholding", th3)
cv2.waitKey(0)

# Finding contour
im2, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print contours
#cnt = contours[4]
cnts = contours

cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break

#cv2.drawContours(im, [cnt], -1, (0,255,0), 3)

## (4) Create mask and do bitwise-op
mask = np.zeros(im.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(im, im, mask=mask)

cv2.imshow("last", im)
cv2.waitKey(0)

sayhello('Akbar')


# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
#keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)




