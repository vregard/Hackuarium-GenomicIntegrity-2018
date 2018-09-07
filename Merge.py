# Standard imports
import numpy as np
import cv2
import imutils
from skimage.measure import compare_ssim
#from skimage.measure import compare_ssim
from termcolor import colored
from matplotlib import pyplot as plt


if __name__ == '__main__':

    #original = cv2.imread('2018-08-02/10-12-46.jpg')
    #original = cv2.imread('2018-08-02/10-13-51.jpg')
    #original = cv2.imread('2018-08-02/10-14-49.jpg')

    #original = cv2.imread('2018-08-03/10-44-21.jpg') # cell ok but nueclus no
    #original = cv2.imread('2018-08-03/10-43-59.jpg')
    original = cv2.imread('2018-08-03/10-43-31.jpg')
    #original = cv2.imread('2018-08-03/10-42-39.jpg')
    #original = cv2.imread('2018-08-03/10-39-23.jpg')

    #backgrd = cv2.imread('2018-08-02/10-14-05.jpg')

    backgrd = cv2.imread('2018-08-03/10-35-31.jpg')

    img1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    backgrd = cv2.cvtColor(backgrd, cv2.COLOR_BGR2GRAY)

    original = cv2.resize(original, (0, 0), fx=0.5, fy=0.5)
    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    backgrd = cv2.resize(backgrd, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('image', img1)
    #cv2.imshow('Background', backgrd)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(img1, backgrd, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    #cv2.imshow("Diff", diff)
    #cv2.waitKey(0)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #cv2.imshow('threshold 1', thresh)

    ret, tmp = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('threshold 2', tmp)

    #cv2.waitKey(0)

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
    #cv2.imshow('src', src)
    img_mask = np.zeros(src.shape, np.uint8)

    theImg = contour_list
    # draw selected contour in bold green
    cv2.polylines(dst, theImg, True, (0, 255, 0), 2)
    # draw the fish into its mask
    cv2.drawContours(img_mask, contour_list, -1, 255, -1)

    #img_mask = cv2.erode(img_mask, None, iterations=2)
    #img_mask = cv2.dilate(img_mask, None, iterations=2)

    #cv2.imshow("Result Mask", img_mask)
    #cv2.imshow("Result Contour", dst)
    #cv2.imshow('src', src)

    output_grey = cv2.bitwise_and(src, src, mask=img_mask)
    #cv2.imshow('output_grey', output_grey)
    src = original
    output_color = cv2.bitwise_and(src, src, mask=img_mask)
    #cv2.imshow('output_color', output_color)

    cv2.waitKey(0)
    cv2.imshow("output_grey", output_grey)
    cv2.imshow("output_color", output_color)
    cv2.waitKey(0)
    ####################################################################################################################
    # marche bien jusqu'a la :)

    #cv2.imwrite('test3.png', output_color)
    #cv2.imwrite('test3.png', output_grey)

    #bgr = cv2.imread(image_path)
    #lab = cv2.cvtColor(output_color, cv2.COLOR_BGR2LAB)
    #lab_planes = cv2.split(lab)
    #gridsize = 8
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    #lab_planes[0] = clahe.apply(lab_planes[0])
    #lab = cv2.merge(lab_planes)
    #bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #cl1 = clahe.apply(output_grey)
    #cv2.imshow('CLAHE', bgr)

    channels = cv2.split(output_color)
    for channel in channels:
        cv2.equalizeHist(channel, channel)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, output_color)
    cv2.imshow('output1.png', output_color)
    cv2.waitKey(0)

    #ycrcb = cv2.cvtColor(output_color, cv2.COLOR_BGR2YCR_CB)
    #cv2.imshow('ycrcb.png', ycrcb)
    #hsv = cv2.cvtColor(output_color, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv.png', hsv)
    #xyz = cv2.cvtColor(output_color, cv2.COLOR_BGR2XYZ)
    #cv2.imshow('xyz.png', xyz)
    #hls = cv2.cvtColor(output_color, cv2.COLOR_BGR2HLS)
    #cv2.imshow('hls.png', hls)
    #cv2.waitKey(0)

    ycrcb = cv2.cvtColor(output_color, cv2.COLOR_BGR2YCR_CB)
    cv2.imshow('ycrcb 1', ycrcb)
    #channels = cv2.split(ycrcb)
    #print len(channels)
    #for channel in channels:
        #cv2.equalizeHist(channel, channel)
    #cv2.equalizeHist(channels[0], channels[0])
    #cv2.merge(channels, ycrcb)
    #cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, output_color)
    #cv2.imshow('output2.png', output_color)
    #cv2.imwrite('ycrcb.png', ycrcb)
    #cv2.waitKey(0)

    #y, cr, cb = ycrcb.split()
    #y.show()

    #cv2.imshow('0.png', ycrcb[:, :, 0])
    #cv2.imshow('1.png', ycrcb[:, :, 1])
    #cv2.imshow('2.png', ycrcb[:, :, 2])
    #cv2.waitKey(0)

    #cv2.imshow('0', ycrcb[0,0,0])
    #cv2.imshow('1', ycrcb[0, 0, 1])
    #cv2.imshow('2', ycrcb[0, 0, 2])

    #ret, tmp = cv2.threshold(ycrcb, 150, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("tmp", tmp)
    #cv2.cvtColor(tmp, cv2.COLOR_YCR_CB2BGR, output_color)
    #cv2.imshow('output3.png', output_color)
    #out_grey = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('out_grey.png', out_grey)

    #im2, contours, hierarchy = cv2.findContours(ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("im2", im2)

    # ycrcb[:, :, 1] # bonne image en noir et blanc
    # ycrcb[:, :, 2]
    src = ycrcb[:, :, 1]

    tmp = ycrcb[:, :, 1]
    blursSize = 17#31  # 27#17
    tmp = cv2.GaussianBlur(tmp, (blursSize, blursSize), 0)
    cv2.imshow('Blur', tmp)

    src = tmp
    nucleus_contour_list = []

    ###
    for i in range(1,len(contour_list)):
        img_m = np.zeros(src.shape, np.uint8)
        cv2.drawContours(img_m, contour_list, i, 255, -1)
        cv2.imshow('mask', img_m)
        out = cv2.bitwise_and(src, src, mask=img_m)
        cv2.imshow('out', out)
        #cv2.waitKey(0)


        mean = cv2.mean(src, mask=img_m)[0]
        ret, tmp = cv2.threshold(out, 255-mean, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('thresh', tmp)

        im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            # if ((len(approx) > 15) & (area > 100) & (area < 1000)):
            if ((area < 1000)):
                # & (area < 1000)):
                nucleus_contour_list.append(contour)


        cv2.waitKey(0)


    ###

    frame = output_color
    cv2.drawContours(frame, nucleus_contour_list, -1, (255, 0, 0), 2)
    cv2.imshow("Nucleus", frame)
    cv2.waitKey(0)


    tmp = ycrcb[:, :, 1]
    blursSize = 31#27#17
    tmp = cv2.GaussianBlur(tmp, (blursSize, blursSize), 0)
    cv2.imshow('Blur', tmp)
    #cv2.imwrite('ycrcb.png', tmp)
    #edge_detected_image = cv2.Canny(tmp, 80, 200)
    #cv2.imshow('Edge', edge_detected_image)

    #bilateral = cv2.bilateralFilter(ycrcb, 10, 75, 75)
    #cv2.imshow('bilateral Blur', bilateral)

    image = tmp

    #kernel = np.ones((2, 2), np.uint8)
    #dilation = cv2.dilate(tmp, kernel, iterations=2)
    #cv2.imshow('dilation', dilation)
    #tmp = dilation


    # Additional Part with Blobs Detection
    #tmp = cv2.bitwise_not(tmp)
    #cv2.imshow("tmp", tmp)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs
    keypoints = detector.detect(tmp)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)

    cv2.waitKey(0)













    # define the list of boundaries
    #lower = [60, 60, 145]; upper = [255, 255, 255]

        #([30, 30, 80], [120, 120, 255])


        #([17, 15, 100], [50, 56, 200]),
        #([86, 31, 4], [220, 88, 50]),
        #([25, 146, 190], [62, 174, 250]),
        #([103, 86, 65], [145, 133, 128])

    # create NumPy arrays from the boundaries
    #lower = np.array(lower, dtype="uint8")
    #upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    #mask = cv2.inRange(image, lower, upper)

    #ret, tmp = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("tmp", tmp)
    #cv2.waitKey(0)

    mask = cv2.inRange(image, 125, 255)
    #cv2.imshow("mask", mask)
    mask = cv2.bitwise_not(mask)
    #out_ycrcb = cv2.bitwise_and(image, image, mask=mask)
    out_grey = cv2.bitwise_and(output_grey, output_grey, mask=mask)

    # show the images
    #cv2.imshow("out_ycrcb", out_ycrcb)
    cv2.imshow("out_grey", out_grey)
    cv2.waitKey(0)

    kernel = np.ones((2, 2), np.uint8)
    # erosion = cv2.erode(output_grey, kernel, iterations=1)
    # cv2.imshow('erosion 1', erosion)
    #output_grey = cv2.bitwise_not(output_grey)
    #cv2.imshow('not out_grey', output_grey)
    erosion = cv2.erode(out_grey, kernel, iterations=3)
    cv2.imshow('erosion 2', erosion)

    dilation = cv2.dilate(erosion, kernel, iterations=2)
    cv2.imshow('dilation', dilation)

    out2 = dilation

    #blursSize = 17
    #tmp = cv2.GaussianBlur(ycrcb, (blursSize, blursSize), 0)
    #cv2.imshow('Blur', tmp)

    #out2 = cv2.cvtColor(tmp, cv2.COLOR_YCR_CB2BGR)
    #cv2.imshow('out2.png', out2)

    #cv2.waitKey(0)

    #output_grey = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('B', output_color[:, :, 0])

    #color = ('b', 'g', 'r')
    #for i, col in enumerate(color):
    #    histr = cv2.calcHist([output_color], [i], None, [256], [0, 256])
    #    plt.plot(histr, color=col)
    #    plt.xlim([0, 256])
    #plt.show()

    #hist = cv2.calcHist([output_color],[0],None,[256],[0,256])
    #plt.hist(output_color.ravel(), 256, [0, 256]); plt.show()

    #full = backgrd - img1
    #cv2.imshow("full", full)
    #out_grey = cv2.bitwise_and(full, full, mask=img_mask)
    #cv2.imshow('out_grey', out_grey)
    #out_grey = cv2.bitwise_not(out_grey)
    #cv2.imshow('not out_grey', out_grey)

    #cv2.imshow('out_grey', output_grey)
    #kernel = np.ones((5, 5), np.uint8)
    #erosion = cv2.erode(output_grey, kernel, iterations=1)
    #cv2.imshow('erosion 1', erosion)

    #output_grey = cv2.bitwise_not(output_grey)
    #cv2.imshow('not out_grey', output_grey)

    #erosion = cv2.erode(output_grey, kernel, iterations=2)
    #cv2.imshow('erosion 2', erosion)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)
    #cv2.imshow('dilation', dilation)

    tmp = dilation

    #blursSize = 17
    #tmp = cv2.GaussianBlur(out_grey, (blursSize, blursSize), 0)
    #tmp = cv2.bitwise_not(tmp)
    #cv2.imshow('tmp', tmp)

    ret, tmp = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', tmp)

    #th2 = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    #cv2.imshow("th2", th2)

    #th3 = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)
    #cv2.imshow("th3", th3)

    frame = output_color
    im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        #if ((len(approx) > 15) & (area > 100) & (area < 1000)):
        if ((area < 1000)):
            # & (area < 1000)):
            contour_list.append(contour)

    cv2.drawContours(frame, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow("Nucleus", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print colored("Number of Cell detected " + str(len(contour_list)), 'green')

