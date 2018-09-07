import matplotlib.pyplot as plt
import sys

#infile = 'GFP5TSS10xPlan1.tif' # change name of file to convert
#infile = 'a_image.tif'
#infile = 'OverlayUSS2_20x.tif'
infile = sys.argv[1]
outfile = infile[:-3]+"jpg"

img = plt.imread(infile)

plt.gray()
plt.axis('off')

plt.imshow(img)
#plt.show()

plt.savefig(outfile)#, "JPEG", quality=100)
