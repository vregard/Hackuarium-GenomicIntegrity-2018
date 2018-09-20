import matplotlib.pyplot as plt
import sys
import os
import cv2

#infile = 'GFP5TSS10xPlan1.tif' # change name of file to convert
#infile = 'a_image.tif'
#infile = 'OverlayUSS2_20x.tif'
file = sys.argv[1]

newpath = file + "-jpg"
os.makedirs(newpath)

for image in os.listdir(file):

    infile = image
    outfile = file + "-jpg/" + infile[:-3]+"jpg"
    infile = file + "/" + image

    img = plt.imread(infile)

    #plt.figure(frameon=False)
    plt.gray()
    plt.axis('off')
    plt.imshow(img)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    #plt.show()
    plt.savefig(outfile)#, transparent = True, bbox_inches='tight', pad_inches=-0.1)
    #plt.savefig(outfile, bbox_inches = 'tight', transparent=True, pad_inches=0)#, "JPEG", quality=100)
