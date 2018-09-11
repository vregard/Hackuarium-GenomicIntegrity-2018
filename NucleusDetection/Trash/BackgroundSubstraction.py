import numpy as np
import cv2

img1 = cv2.imread('2018-08-02/10-12-46.jpg',0)
backgrd = cv2.imread('2018-08-02/10-14-05.jpg',0)

cv2.imshow('image', img1)
cv2.imshow('Background', backgrd)

result = img1 - backgrd
cv2.imshow('result', result)

small = cv2.resize(result, (0,0), fx=0.5, fy=0.5)
small = cv2.bitwise_not(small)
cv2.imshow('res', small)
cv2.waitKey(0)

cv2.imwrite('test.png',small)