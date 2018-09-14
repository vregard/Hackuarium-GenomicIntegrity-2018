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
import numpy
import shutil

file = sys.argv[1] # name of the folder
nbrImages = 4 # number of last image
formatImage = "jpg" # format of the images
interm = False#True
nbrVerticalImages = 2

newfile = file + "-tmp"
os.makedirs(newfile)

for j in xrange(1, nbrImages + 1): # stitch in horizontal direction

	i = j % nbrVerticalImages
	#print i

	# si 1 on fait rien
	# si 2 conserver le 2 actuel
	# si 3,.... traitement normal
		# si 0 enregistrer l'image resultante

	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	if (i == 2 or (nbrVerticalImages == 2 and i == 0)):
		n = j - 1
		first = file + "/" + str(n) + "." + formatImage
		imageA = cv2.imread(first)
		second = file + "/" + str(j) + "." + formatImage
		imageB = cv2.imread(second)

		imageA = imutils.resize(imageA, width=400)
		imageB = imutils.resize(imageB, width=400)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		(result, vis) = stitcher.stitch_horizontal([imageA, imageB], showMatches=True)

	elif i != 1:
		imageA = result
		second = file + "/" + str(j) + "." + formatImage
		imageB = cv2.imread(second)

		imageB = imutils.resize(imageB, width=400)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		(result, vis) = stitcher.stitch_horizontal([imageA, imageB], showMatches=True)

	if interm:
		# show the images
		cv2.imshow("Image A", imageA)
		cv2.imshow("Image B", imageB)
		cv2.imshow("Keypoint Matches", vis)
		cv2.imshow("Result", result)
		cv2.waitKey(0)

	if i == 0:
		n = int(j/nbrVerticalImages)
		outfile = newfile + "/completePanorama" + str(n) + ".png"
		cv2.imwrite(outfile, result)

		#cv2.imshow("Result", result)
		#cv2.waitKey(0)


nbrPanorama = nbrImages/nbrVerticalImages
#interm = True

for i in xrange(1, nbrPanorama + 1): # stitch in vertical direction

	#print "nbrPanorama"

	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	if (i == 2):

		first = newfile + "/completePanorama1.png"
		imageA = cv2.imread(first)
		second = newfile + "/completePanorama" + str(i) + ".png"
		imageB = cv2.imread(second)

		imageA = numpy.rot90(imageA, 1)
		imageB = numpy.rot90(imageB, 1)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		(result, vis) = stitcher.stitch_horizontal([imageA, imageB], showMatches=True)

	elif i > 2:
		imageA = result
		second = newfile + "/completePanorama" + str(i) + ".png"
		imageB = cv2.imread(second)

		imageB = numpy.rot90(imageB, 1)

		# stitch the images together to create a panorama
		stitcher = Stitcher()
		(result, vis) = stitcher.stitch_vertical([imageA, imageB], showMatches=True)

	if interm:
		# show the images
		cv2.imshow("Image A", imageA)
		cv2.imshow("Image B", imageB)
		cv2.imshow("Keypoint Matches", vis)
		cv2.imshow("Result", result)
		cv2.waitKey(0)

	if i == nbrPanorama:
		result = numpy.rot90(result, 3)

		cv2.imwrite("finalPanorama.png", result)

		cv2.imshow("Final Result", result)
		cv2.waitKey(0)

		shutil.rmtree(newfile)



# python stitch.py --first PanoramaJennifer-jpg/1.jpg --second PanoramaJennifer-jpg/2.jpg