# USAGE
# python rotate.py --image grand_canyon.png

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# grab the dimensions of the image and calculate the center of the image
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)

# rotate our image by 45 degrees
# M = cv2.getRotationMatrix2D((cX, cY), -30, 1.0)
# M = cv2.getRotationMatrix2D((cX, cY), 110, 1.0)
M = cv2.getRotationMatrix2D((50, 50), 88, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 30 Degrees", rotated)
bgr=rotated[10, 10]
# bgr=rotated[254, 335]
# bgr=rotated[136, 312]
print(bgr)
cv2.waitKey(0)
