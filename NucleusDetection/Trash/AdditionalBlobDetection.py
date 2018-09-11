# Standard imports
import numpy as np
import cv2

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
    #cv2.imwrite("../img/BrightnessPreprocess.png", tmp)





    #bilateral_filtered_image = cv2.bilateralFilter(tmp, 5, 5000, 5000)
    #cv2.imshow('Bilateral', bilateral_filtered_image)
    #src = bilateral_filtered_image



    ## global thresholding
    ret, tmp = cv2.threshold(tmp, th1, 255, cv2.THRESH_BINARY_INV)
    # adaptative mean thresholding
    #th2 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    # adaptative gaussian thresholding
    #tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 2)

    # threshold to select dark tuna
    #ret, tmp = cv2.threshold(tmp, th1, 255, cv2.THRESH_BINARY_INV)
    #ret, tmp = cv2.threshold(tmp, th1, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', tmp)

    # find external contours ignores holes in the fish
    im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dst = im.copy()
    #dst = src

    # print(contours)




# Additional Part with Blobs Detection
    #tmp = cv2.bitwise_not(tmp)
    # Set up the detector with default parameters.
    #detector = cv2.SimpleBlobDetector_create()

    # Detect blobs
    #keypoints = detector.detect(tmp)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, np.array([]), (0, 0, 255),
    #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    #cv2.imshow("Keypoints", im_with_keypoints)




# edge detection
    #edge_detected_image = cv2.Canny(tmp, 75, 200)
    #cv2.imshow('Edge', edge_detected_image)




    # find external contours ignores holes in the fish
    #im2, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #dst = im.copy()
    #dst = src


    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 10000) & (area < 20000)):
            contour_list.append(contour)

    # Display result
    cv2.drawContours(im, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', im)




    #maxDim = 0
    largest = -1

    for i in range(len(contours)):
        # draw all contours in red
        cv2.drawContours(dst, contours, largest, (0, 0, 255), 1)
        #dim = len(contours[i]) # area is more accurate but more expensive
        #if (dim > maxDim):
        #    maxDim = dim
        #    largest = i


    # The tuna as binary mask
    #cv::Mat fishMask = cv::Mat::zeros(src.size(), CV_8UC1)
    ##The tuna as contour
    #vector<cv::Point> theFish
    img_mask = np.zeros(src.shape, np.uint8)

    #if (largest >= 0):
        #theImg = contours[largest]
        # draw selected contour in bold green
        #cv2.polylines(dst, theImg, True, (0, 255,0), 2)
        # draw the fish into its mask
        #cv2.drawContours(img_mask, contours, largest, 255, -1)

    theImg = contour_list
    # draw selected contour in bold green
    cv2.polylines(dst, theImg, True, (0, 255, 0), 2)
    # draw the fish into its mask
    cv2.drawContours(img_mask, contour_list, largest, 255, -1)

    cv2.imshow("Result Mask", img_mask)
    cv2.imshow("Result Contour", dst)
    #cv2.imwrite("../img_mask.png", img_mask)
    #cv2.imwrite("../result.png", dst)


if __name__ == '__main__':

    #src = cv2.imread("Vio4.PNG")
    src = cv2.imread("test.png")
    cv2.imshow(winName, src)

   # bilateral_filtered_image = cv2.bilateralFilter(src, 5, 300, 300)
   # cv2.imshow('Bilateral', bilateral_filtered_image)
   # src = bilateral_filtered_image


    # edge detection
    #edge_detected_image = cv2.Canny(src, 175, 200)
    #cv2.imshow('Edge start', edge_detected_image)


    # Set up the detector with default parameters.
    #detector = cv2.SimpleBlobDetector_create()

    # Detect blobs
    #keypoints = detector.detect(src)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(src, keypoints, np.array([]), (0, 0, 255),
    #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    #cv2.imshow("Keypoints", im_with_keypoints)




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
    th1 = int(30 * 255 / 100)  # tuna is dark than select dark zone below 33% of full range

    cv2.createTrackbar("Equalize", winName, useEqualize, 1, onTunaFishTrackbar)
    cv2.createTrackbar("Blur Sigma", winName, blursSize, 100, onTunaFishTrackbar)
    cv2.createTrackbar("Threshold", winName, th1, 255, onTunaFishTrackbar)

    while(1):
        #--- Using cv2.getTrackbarPos() to get values from the slider ---
        useEqualize = cv2.getTrackbarPos('Equalize', winName)
        blurSize = cv2.getTrackbarPos('Blur Sigma', winName)
        th1 = cv2.getTrackbarPos('Threshold', winName)

        #if len(brightness) > 0:
        #    print("brightness ok")

        #if len(src) > 0:
        #    print("src ok")

        onTunaFishTrackbar(src, brightness, useEqualize, blurSize, th1)

        #--- Press Q to quit ---
        k = cv2.waitKey(1) #& 0xFF
        if k == 27:
            break
            #cv2.waitKey(0)


    #onTunaFishTrackbar(src, 0, 0, useEqualize, blursSize, th1, brightness=brightness)

    #cv2.waitKey(0)
    cv2.destroyAllWindows()

