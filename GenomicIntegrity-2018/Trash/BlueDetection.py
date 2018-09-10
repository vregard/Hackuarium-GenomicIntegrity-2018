# Standard imports
import numpy as np
import cv2
#from skimage.measure import compare_ssim

#winName = "Cell fish"

masking = True
#tracking = True

def nothing(*arg):
    pass

def setBlue():
    #cv2.setTrackbarPos('lower - red', 'RGB', 80)
    #cv2.setTrackbarPos('lower - green', 'RGB', 30)
    #cv2.setTrackbarPos('lower - blue', 'RGB', 30)

    #cv2.setTrackbarPos('upper - red', 'RGB', 255)
    #cv2.setTrackbarPos('upper - green', 'RGB', 120)
    #cv2.setTrackbarPos('upper - blue', 'RGB', 120)


    #cv2.setTrackbarPos('lower - red', 'RGB', 145)
    #cv2.setTrackbarPos('lower - green', 'RGB', 60)
    #cv2.setTrackbarPos('lower - blue', 'RGB', 60)

    #cv2.setTrackbarPos('upper - red', 'RGB', 255)
    #cv2.setTrackbarPos('upper - green', 'RGB', 255)
    #cv2.setTrackbarPos('upper - blue', 'RGB', 255)


    cv2.setTrackbarPos('lower - red', 'RGB', 125)
    cv2.setTrackbarPos('lower - green', 'RGB', 60)
    cv2.setTrackbarPos('lower - blue', 'RGB', 60)

    cv2.setTrackbarPos('upper - red', 'RGB', 255)
    cv2.setTrackbarPos('upper - green', 'RGB', 113)
    cv2.setTrackbarPos('upper - blue', 'RGB', 113)

if __name__ == '__main__':

    #img1 = cv2.imread('2018-08-02/10-12-46.jpg')
    #img1 = cv2.imread('2018-08-02/10-13-51.jpg')
    #img1 = cv2.imread('2018-08-02/10-14-49.jpg')

    small = cv2.imread('ycrcb.png')

    #cv2.imshow('image', img1)
    #color = img1

    #small = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)
    #small = cv2.bitwise_not(small)
    cv2.imshow('res', small)

    image = small

    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imshow('RGB', small)

    cv2.createTrackbar("lower - red", 'RGB', 10, 255, nothing)
    cv2.createTrackbar("lower - green", 'RGB', 10, 255, nothing)
    cv2.createTrackbar('lower - blue', 'RGB', 10, 255, nothing)

    cv2.createTrackbar('upper - red', 'RGB', 11, 255, nothing)
    cv2.createTrackbar('upper - green', 'RGB', 11, 255, nothing)
    cv2.createTrackbar('upper - blue', 'RGB', 11, 255, nothing)

    setBlue()

    if masking:
        while(1):
            # Capture frame-by-frame
            #ret, image = cap.read()

            thrs1 = cv2.getTrackbarPos('lower - red', 'RGB')
            thrs2 = cv2.getTrackbarPos('lower - green', 'RGB')
            thrs3 = cv2.getTrackbarPos('lower - blue', 'RGB')

            thrs4 = cv2.getTrackbarPos('upper - red', 'RGB')
            thrs5 = cv2.getTrackbarPos('upper - green', 'RGB')
            thrs6 = cv2.getTrackbarPos('upper - blue', 'RGB')

            #print(thrs1)

            #if (thrs1 > thrs4):
            #    cv2.setTrackbarPos('lower - red', 'RGB', thrs4 - 1)
            #if (thrs2 > thrs5):
            #    cv2.setTrackbarPos('lower - green', 'RGB', thrs5 - 1)
            #if (thrs3 > thrs6):
            #    cv2.setTrackbarPos('lower - blue', 'RGB', thrs6 - 1)

            # define the list of boundaries
            boundaries = [
                ([thrs3, thrs2, thrs1], [thrs6, thrs5, thrs4])
            ]

            # loop over the boundaries
            for (lower, upper) in boundaries:
                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")

                # find the colors within the specified boundaries and apply
                # the mask
                mask = cv2.inRange(image, lower, upper)
                cv2.imshow('mask', mask)
                output = cv2.bitwise_and(image, image, mask=mask)

                #imageOut = np.hstack([image, output])

            # Display the resulting frame
            cv2.imshow('result', np.hstack([image, output]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                masking = False
                break

        # When everything done, release the capture
        #cap.release()
        cv2.destroyAllWindows()




