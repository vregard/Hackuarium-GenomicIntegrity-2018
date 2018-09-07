import numpy as np
import cv2
import imutils
from skimage.measure import compare_ssim

# img1 = cv2.imread('2018-08-02/10-12-46.jpg')#, 0)
img1 = cv2.imread('2018-08-02/10-13-51.jpg')#, 0)
#img1 = cv2.imread('2018-08-02/10-14-49.jpg')#, 0)
backgrd = cv2.imread('2018-08-02/10-14-05.jpg')#,0)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
backgrd = cv2.cvtColor(backgrd, cv2.COLOR_BGR2GRAY)

img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
backgrd = cv2.resize(backgrd, (0, 0), fx=0.5, fy=0.5)

cv2.imshow('image', img1)
cv2.imshow('Background', backgrd)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(img1, backgrd, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
#for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    #(x, y, w, h) = cv2.boundingRect(c)
    #cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.rectangle(backgrd, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
#cv2.imshow("Original", img1)
#cv2.imshow("Modified", backgrd)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

#result = img1 - backgrd
#cv2.imshow('result', result)

#small = cv2.resize(result, (0,0), fx=0.5, fy=0.5)
#small = cv2.bitwise_not(small)
#cv2.imshow('res', small)
#cv2.waitKey(0)

#cv2.imwrite('test.png',small)