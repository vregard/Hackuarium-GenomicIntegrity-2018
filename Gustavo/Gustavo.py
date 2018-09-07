# Standard imports
import numpy as np
import cv2
import imutils
#from skimage.measure import compare_ssim
#from skimage.measure import compare_ssim
from termcolor import colored


if __name__ == '__main__':

    original = cv2.imread('IMG_20180626_105547.jpg')

    img1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    original = cv2.resize(original, (0, 0), fx=0.2, fy=0.2)
    img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2)

    cv2.imshow('original', original)
    cv2.imshow('image', img1)

    diff = img1
    tmp = diff

    #cv2.imshow("Diff", diff)
    cv2.waitKey(0)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #cv2.imshow('threshold 1', thresh)

    # Detection boxes
    boxes_img = diff
    #kernel = np.ones((1, 1), np.uint8)
    #opening = cv2.morphologyEx(boxes_img, cv2.MORPH_OPEN, kernel)
    #blursSize = 15
    #boxes_img = cv2.GaussianBlur(boxes_img, (blursSize, blursSize), 0)
    #cv2.imshow('blursSize', boxes_img)
    #tmp = boxes_img
    #boxes_img = cv2.bitwise_not(boxes_img)
    #cv2.imshow('inv blursSize', boxes_img)
    #tmp = cv2.adaptiveThreshold(boxes_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    #ret, tmp = cv2.threshold(boxes_img, 150, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('threshold 1', tmp)

    #ret, tmp = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('threshold 2', tmp)

    #cv2.waitKey(0)

    #im = cv2.bitwise_not(tmp)
    im = tmp.copy()
    dst = img1.copy()

    #cv2.imshow('im', im)

    edge_detected_image = cv2.Canny(im, 80, 200)
    cv2.imshow('Edge', edge_detected_image)
    cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edge_detected_image, kernel, iterations=1)
    cv2.imshow('dilation', dilation)

    #erosion = cv2.erode(dilation, kernel, iterations=1)
    #cv2.imshow('erosion', erosion)

    # find external contours ignores holes in the fish
    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((area > 1000)):
        #if ((len(approx) > 8) & (area > 10000)): #& (area < 20000)):
        #if ((len(approx) > 8)):# & (area < 20000)):
            contour_list.append(contour)

    # Display result
    cv2.drawContours(dst, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', dst)

    #cv2.waitKey(0)

    for i in range(len(contours)):
        # draw all contours in red
        cv2.drawContours(dst, contours, -1, (0, 0, 255), 1)

    # Binary mask
    src = img1
    img_mask = np.zeros(src.shape, np.uint8)

    theImg = contour_list
    # draw selected contour in bold green
    cv2.polylines(dst, theImg, True, (0, 255, 0), 2)
    # draw the fish into its mask
    cv2.drawContours(img_mask, contour_list, -1, 255, -1)
    im2, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(dst, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected 2', dst)

    #img_mask = cv2.erode(img_mask, None, iterations=2)
    #img_mask = cv2.dilate(img_mask, None, iterations=2)

    cv2.imshow("Result Mask", img_mask)
    #cv2.imshow("Result Contour", dst)
    #cv2.imshow('src', src)

    output_grey = cv2.bitwise_and(src, src, mask=img_mask)
    #cv2.imshow('output_grey', output_grey)
    src = original
    output_color = cv2.bitwise_and(src, src, mask=img_mask)
    #cv2.imshow('output_color', output_color)

    cv2.waitKey(0)
    # marche bien jusqu'a la :)




