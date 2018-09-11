# Standard imports
import numpy as np
import cv2
#from skimage.measure import compare_ssim

winName = "Cell fish"

def onTunaFishTrackbar(im, brightness, useEqualize=1, blursSize=21, th1=None):
    winName = "Cell fish"
    tmp = brightness

    if (blursSize >= 3):
        blursSize += (1 - blursSize % 2)
        tmp = cv2.GaussianBlur(tmp, (blursSize, blursSize), 0)

    if (useEqualize):
        tmp = cv2.equalizeHist(tmp)

    cv2.imshow("Brightness Preprocess", tmp)


    # global thresholding
    ret, tmp = cv2.threshold(tmp, th1, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('threshold', tmp)

    # find external contours ignores holes in the fish
    im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dst = im.copy()

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 10000) & (area < 20000)):
        #if ((area > 10000)):# & (area < 20000)):
            contour_list.append(contour)

    # Display result
    cv2.drawContours(im, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', im)

    largest = -1

    for i in range(len(contours)):
        # draw all contours in red
        cv2.drawContours(dst, contours, largest, (0, 0, 255), 1)

    # Binary mask
    img_mask = np.zeros(src.shape, np.uint8)

    theImg = contour_list
    # draw selected contour in bold green
    cv2.polylines(dst, theImg, True, (0, 255, 0), 2)
    # draw the fish into its mask
    cv2.drawContours(img_mask, contour_list, largest, 255, -1)

    cv2.imshow("Result Mask", img_mask)
    cv2.imshow("Result Contour", dst)


if __name__ == '__main__':

    #img1 = cv2.imread('2018-08-02/10-12-46.jpg', 0)
    img1 = cv2.imread('2018-08-02/10-13-51.jpg', 0)
    #img1 = cv2.imread('2018-08-02/10-14-49.jpg', 0)

    backgrd = cv2.imread('2018-08-02/10-14-05.jpg', 0)

    cv2.imshow('image', img1)
    cv2.imshow('Background', backgrd)

    result = img1 - backgrd

    #result = cv2.imread('test3.png', 0)
    cv2.imshow('result', result)

    #result2 = cv2.subtract(img1, backgrd)
    #result2 = cv2.bitwise_not(result2)
    #cv2.imshow('result2', result2)

    #(score, diff) = compare_ssim(img1, backgrd,full=True)
    #cv2.imshow('diff', diff)

    #result = result2

    small = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
    small = cv2.bitwise_not(small)
    cv2.imshow('res', small)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite('test3.png', small)

    #src = cv2.imread("Vio4.PNG")
    src = cv2.imread("test3.png")
    cv2.imshow(winName, src)

    dst = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(dst)
    #hue = hsv_planes[0]
    #saturation = hsv_planes[1]
    brightness = hsv_planes[2]

    # default settings for params
    useEqualize = 1
    #blursSize = 21
    #th1 = int(33.0 * 255 / 100) # tuna is dark than select dark zone below 33% of full range
    blursSize = 10
    #th1 = int(55 * 255 / 100)  # tuna is dark than select dark zone below 33% of full range
    # for the first trial 30 is good
    #th1 = int(30 * 255 / 100)  # tuna is dark than select dark zone below 33% of full range
    th1 = int(40 * 255 / 100)  # tuna is dark than select dark zone below 33% of full range

    cv2.createTrackbar("Equalize", winName, useEqualize, 1, onTunaFishTrackbar)
    cv2.createTrackbar("Blur Sigma", winName, blursSize, 100, onTunaFishTrackbar)
    cv2.createTrackbar("Threshold", winName, th1, 255, onTunaFishTrackbar)

    while(1):
        #--- Using cv2.getTrackbarPos() to get values from the slider ---
        useEqualize = cv2.getTrackbarPos('Equalize', winName)
        blurSize = cv2.getTrackbarPos('Blur Sigma', winName)
        th1 = cv2.getTrackbarPos('Threshold', winName)

        onTunaFishTrackbar(src, brightness, useEqualize, blurSize, th1)

        #--- Press Q to quit ---
        k = cv2.waitKey(1) #& 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

