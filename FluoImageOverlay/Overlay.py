# GOAL: overlay image + fluo image
import numpy as np
import cv2
import sys

if __name__ == '__main__':

    #fluo = cv2.imread('2018-08-15/16-04-02.jpg')
    #img = cv2.imread('2018-08-15/16-04-17.jpg')

    #fluo = cv2.imread('2018-08-15/16-28-53.jpg')
    #original = cv2.imread('2018-08-15/16-29-06.jpg')

    #fluo = cv2.imread('2018-08-15/17-12-32.jpg')
    #original = cv2.imread('2018-08-15/17-12-10.jpg')

    fluo = cv2.imread(sys.argv[1])
    original = cv2.imread(sys.argv[2])

    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    fluo = cv2.resize(fluo, (0, 0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    #b = 90.  # brightness
    #c = 90.  # contrast
    #fluo = cv2.addWeighted(fluo, 1. + c / 127., fluo, 0, b - c)
    #cv2.imshow('Fluo - aft', fluo)

    #dst = cv2.addWeighted(img,0.5, fluo, 1, 0)
    dst = cv2.addWeighted(img, 1, fluo, 2, 0)
    cv2.imshow('Overlay', dst)

    cv2.waitKey(0)