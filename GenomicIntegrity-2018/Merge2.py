# Standard imports
import numpy as np
import cv2
#import imutils
from skimage.measure import compare_ssim
from termcolor import colored
from matplotlib import pyplot as plt

# for test
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float

#def toucheImageBorder( contour, imageSize ):
#    xMin = 0
#    yMin = 0
#    xMax = imageSize.width - 1
#    yMax = imageSize.height - 1

    # Use less / greater comparisons to potentially support contours outside of
    # image coordinates, possible future workarounds with cv::copyMakeBorder where
    # contour coordinates may be shifted and just to be safe.
#    if (bb.x <= xMin || bb.y <= yMin || bb.width >= xMax || bb.height >= yMax):
#        return True
#    else:
#        return False

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
    cv2.imshow('diff', diff)

    ret, tmp = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('threshold 2', tmp)

    # find external contours ignores holes in the fish
    im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Finding circular shapes
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 10000)): #& (area < 20000)):
            contour_list.append(contour)

    # Binary mask
    src = img1
    #cv2.imshow('src', src)
    img_mask = np.zeros(src.shape, np.uint8)
    cv2.drawContours(img_mask, contour_list, -1, 255, -1)
    #cv2.imshow("Bef Result Mask", img_mask)

    # check de l'aire des bouts - masque
##
    not_mask = cv2.bitwise_not(img_mask)
    #cv2.imshow("not_mask", not_mask)
    im2, contours, hierarchy = cv2.findContours(not_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            contour_list.append(contour)

    img_mask = np.zeros(src.shape, np.uint8)
    cv2.drawContours(img_mask, contour_list, -1, 255, -1)
    cv2.imshow("Aft Result Mask", img_mask)

    # Find new contour => complete contour
    im2, final_contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dst = img1.copy()
    cv2.drawContours(dst, final_contours, -1, (255, 0, 0), 2)
    cv2.imshow('Final Objects Detected', dst)
##

    output_grey = cv2.bitwise_and(src, src, mask=img_mask)
    src = original
    output_color = cv2.bitwise_and(src, src, mask=img_mask)

    #cv2.imshow("output_grey", output_grey)
    #cv2.imshow("output_color", output_color)
    #cv2.waitKey(0)
    ####################################################################################################################
    # marche bien jusqu'a la :)

    channels = cv2.split(output_color)
    for channel in channels:
        cv2.equalizeHist(channel, channel)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, output_color)
    #cv2.imshow('output1.png', output_color)
    #cv2.waitKey(0)

    ycrcb = cv2.cvtColor(output_color, cv2.COLOR_BGR2YCR_CB)
    #cv2.imshow('ycrcb', ycrcb)

    #cv2.imshow('0.png', ycrcb[:, :, 0])
    cv2.imshow('1.png', ycrcb[:, :, 1]) # bonne image en noir et blanc
    cv2.imshow('2.png', ycrcb[:, :, 2])
    cv2.waitKey(0)

    ycrcb_1 = ycrcb[:, :, 1]
    blursSize = 17 #13 #31 # 27 #17
    tmp = cv2.GaussianBlur(ycrcb_1, (blursSize, blursSize), 0)
    cv2.imshow('Blur', tmp)


#
    ycrcb_2 = ycrcb[:, :, 2]
    blursSize = 17  # 13 #31 # 27 #17
    tmp2 = cv2.GaussianBlur(ycrcb_2, (blursSize, blursSize), 0)
    cv2.imshow('Blur', tmp2)

    im = tmp2

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=20)

    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()

    plt.show()
#


    src = tmp
    nucleus_contour_list = []
    all_contour_list = []

    ###
    contour_list = final_contours
    print len(contour_list)
    for i in range(0,len(contour_list)): # inside each object (group of cells)
        img_m = np.zeros(src.shape, np.uint8)
        cv2.drawContours(img_m, contour_list, i, 255, -1)
        # cv2.imshow('mask', img_m)

    #
        im = cv2.bitwise_and(tmp2, tmp2, mask=img_m)
        cv2.imshow('out 2', im)

        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=20, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance=20)

        # display results
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(im, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')

        ax[2].imshow(im, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')

        fig.tight_layout()

        plt.show()
    #


        out = cv2.bitwise_and(src, src, mask=img_m)
        cv2.imshow('out', out)
        #cv2.waitKey(0)

        equ = cv2.equalizeHist(out)
        cv2.imshow('equalized', equ)
        out = equ

        mean = cv2.mean(src, mask=img_m)[0]
        print mean

        hist_mask = cv2.calcHist([src], [0], img_m, [256], [0, 256])
        #min_value, max_value, min_idx, max_idx = cv2.GetMinMaxHistValue(hist_mask)
        #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(hist_mask)
        #print minVal

        a = np.arange(0, 256) #a = range(1, 256)
        a = np.array([a])
        a = a.T

        tmp_arr = np.append(hist_mask,a, axis=1)

        sel = np.where(tmp_arr[:,0] != 0)
        tmp_array = tmp_arr[sel]

        tot = sum(tmp_array[:,0])
        tmp_array[:, 0] = tmp_array[:,0]/tot

        percentageInterest = 0.001#1 # take first 20%
        #maximum = 0.2 * tot
        i = 0
        summed = 0
        while summed < percentageInterest:
            i = i + 1
            summed = sum(tmp_array[1:i, 0])



        thresh = tmp_array[i,1]
        print thresh

        #plt.plot(hist_mask);
        #plt.show()


        #cv.ThreshHist(hist, threshold)

        #mean = cv2.mean(equ, mask=img_m)[0]
        #print mean

        #hist_mask = cv2.calcHist([equ], [0], img_m, [256], [0, 256])
        #plt.plot(hist_mask); plt.show()


        #plt.hist(out.ravel(), 256, [0, 256]); plt.show()

        if mean > 137:
            thresh = 5#13
        elif mean > 132:
            thresh = 20
        #elif mean > 130:
        #    thresh = 13
        elif mean > 127:
            thresh = 13
        elif mean > 125:
            thresh = 10
        elif mean > 119:
            thresh = 13
        else:
            thresh = 3

        #ret, tmp = cv2.threshold(out, 255 - mean, 255, cv2.THRESH_BINARY_INV)
        #ret, tmp = cv2.threshold(out, 6, 255, cv2.THRESH_BINARY_INV) # ou 15
        #ret, tmp = cv2.threshold(out, 15, 255, cv2.THRESH_BINARY_INV)  # ou 15
        ret, tmp = cv2.threshold(out, thresh, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('thresh 1', tmp)
        kernel = np.ones((2, 2), np.uint8)
        #dilation = cv2.dilate(tmp, kernel, iterations=3)
        erosion = cv2.erode(tmp, kernel, iterations=4)
        dilation = cv2.dilate(erosion, kernel, iterations=3)
        # cv2.imshow('dilation', dilation)
        tmp = dilation
        cv2.imshow('thresh 2', tmp)

        im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        nucleus = False
        new_cell_contour_list = []
        new_nucleus_contour_list = []
        new_all_contour_list = []

        for i in range(0,len(contours)): # for each object detected inside the cell
            contour = contours[i]
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)

            # if ((len(approx) > 15) & (area > 100) & (area < 1000)):
            if ((area > 5000) & (area < 510000)):
                new_cell_contour_list.append(contour)

        #if len(new_cell_contour_list) > 1:
            # considerer un objet a la fois

        #for contour in contours:
            #approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            #area = cv2.contourArea(contour)

            if ((area > 50) & (area < 1000)): # threshold to detect only nucleus
                # compute the center of the contour
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                nucleus_model = np.zeros(src.shape, np.uint8)
                cv2.circle(nucleus_model, (cX, cY), 20, (255, 0, 0), -1)
                #cv2.imshow("nucleus_model", nucleus_model)

                img_m = np.zeros(src.shape, np.uint8)
                cv2.drawContours(img_m, contours, i, 255, -1)
                #cv2.imshow('nucleus candidate', img_m)

                score = cv2.matchShapes(contour, nucleus_model, 1, 0.0) # compare the candidate nucleus with circle
                print ("score: " + str(score))

                if (score < 1.5): # 1.6 for certain images
                    nucleus = True
                    new_nucleus_contour_list.append(contour)

            if ((area < 1000)): # remember contour + nucleus that may be nucleus or micronucleii
                new_all_contour_list.append(contour)

        if nucleus == True: # remove cells without visible nucleus from the count
            for contour in new_nucleus_contour_list:
                nucleus_contour_list.append(contour)
                # check: micronucleii within a circle of 3*nucleus_diameter around the nuclei

                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                small_mask = np.zeros(src.shape, np.uint8)
                cv2.circle(small_mask, (cX, cY), 30, (255, 0, 0), -1)
                #cv2.putText(image, "center", (cX - 20, cY - 20),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # show the image
                # cv2.imshow("small_mask", small_mask)
                out2 = cv2.bitwise_and(ycrcb_1, ycrcb_1, mask=small_mask)
                cv2.imshow('out2', out2)
                cv2.waitKey(0)

            for contour in new_all_contour_list:
                all_contour_list.append(contour)

        print len(new_cell_contour_list)

        cv2.waitKey(0)


    ###

    frame = output_color
    #cv2.drawContours(frame, all_contour_list, -1, (255, 0, 0), 2)
    cv2.drawContours(frame, all_contour_list, -1, (255, 0, 0), 2)
    cv2.drawContours(frame, nucleus_contour_list, -1, (0, 255, 0), 2)
    cv2.imshow("Nucleus", frame)
    cv2.waitKey(0)

    print colored("Number of Cell detected " + str(len(nucleus_contour_list)), 'green')

