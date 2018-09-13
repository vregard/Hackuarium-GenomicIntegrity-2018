# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# Base of the code is taken from https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# And then re-arranged to allow the stitching of all the images present in a file

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import sys
import os

file = sys.argv[1] # name of the folder
maxImage = 4 # number of last image
formatImage = "jpg" # format of the images
interm = False#True

for i in xrange(2, maxImage+1):

	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	if i == 2:
		first = file + "/1." + formatImage
		imageA = cv2.imread(first)
		second = file + "/" + str(i) + "." + formatImage
		imageB = cv2.imread(second)

		imageA = imutils.resize(imageA, width=400)
		imageB = imutils.resize(imageB, width=400)

	else:
		imageA = result
		second = file + "/" + str(i) + "." + formatImage
		imageB = cv2.imread(second)

		imageB = imutils.resize(imageB, width=400)

	# stitch the images together to create a panorama
	stitcher = Stitcher()
	(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

	if interm:
		# show the images
		cv2.imshow("Image A", imageA)
		cv2.imshow("Image B", imageB)
		cv2.imshow("Keypoint Matches", vis)
		cv2.imshow("Result", result)
		cv2.waitKey(0)

cv2.imwrite('completePanorama.png',result)

cv2.imshow("Final Result", result)
cv2.waitKey(0)


# python stitch.py --first PanoramaJennifer-jpg/1.jpg --second PanoramaJennifer-jpg/2.jpg