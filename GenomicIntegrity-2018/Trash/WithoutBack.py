# Standard imports
import numpy as np
import cv2
import imutils
#from skimage.measure import compare_ssim
#from skimage.measure import compare_ssim
from termcolor import colored


if __name__ == '__main__':

    #original = cv2.imread('2018-08-02/10-12-46.jpg')
    #original = cv2.imread('2018-08-02/10-13-51.jpg')
    original = cv2.imread('2018-08-02/10-14-49.jpg')
    original = cv2.imread('more_info/IMG_1394.JPG')

    #backgrd = cv2.imread('2018-08-02/10-14-05.jpg')

    img1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #backgrd = cv2.cvtColor(backgrd, cv2.COLOR_BGR2GRAY)

    original = cv2.resize(original, (0, 0), fx=0.2, fy=0.2)
    img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2)
    #backgrd = cv2.resize(backgrd, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('image', img1)
    #cv2.imshow('Background', backgrd)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    #(score, diff) = compare_ssim(img1, backgrd, full=True)
    #diff = (diff * 255).astype("uint8")
    diff = img1
    #print("SSIM: {}".format(score))

    cv2.imshow("Diff", diff)
    cv2.waitKey(0)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #cv2.imshow('threshold 1', thresh)

    ret, tmp = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('threshold 2', tmp)

    cv2.waitKey(0)

    im = tmp.copy()
    #cv2.imshow('im', im)

    dst = img1.copy()
    #cv2.imshow('dst', dst)

    # find external contours ignores holes in the fish
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 10000)): #& (area < 20000)):
        #if ((area > 10000)):# & (area < 20000)):
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

    #cv2.imwrite('test3.png', output_color)
    #cv2.imwrite('test3.png', output_grey)

    full = backgrd - img1
    #cv2.imshow("full", full)
    out_grey = cv2.bitwise_and(full, full, mask=img_mask)
    #cv2.imshow('out_grey', out_grey)
    out_grey = cv2.bitwise_not(out_grey)
    #cv2.imshow('not out_grey', out_grey)

    blursSize = 17
    tmp = cv2.GaussianBlur(out_grey, (blursSize, blursSize), 0)
    tmp = cv2.bitwise_not(tmp)
    #cv2.imshow('tmp', tmp)

    ret, tmp = cv2.threshold(tmp, 45, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', tmp)

    frame = output_color
    im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if ((area > 100) & (area < 1000)):
            contour_list.append(contour)

    cv2.drawContours(frame, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow("Nucleus", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print colored("Number of Cell detected " + str(len(contour_list)), 'green')

