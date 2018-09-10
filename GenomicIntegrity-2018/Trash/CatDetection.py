import numpy as np
import cv2 as cv

# Read image
# Load an color image in grayscale
img = cv.imread('chaton1.jpeg',0)
# Test if the read was done correctly
print img

# Display image
cv.imshow('image', img)
cv.waitKey(0)
#cv.destroyAllWindows()

# Save an image
cv.imwrite('messigray.png',img)

