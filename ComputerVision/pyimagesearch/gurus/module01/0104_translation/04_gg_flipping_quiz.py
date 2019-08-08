# USAGE
# python flipping.py --image florida_trip.png

# import the necessary packages
import argparse
import cv2
import imutils

HORIZ = 1
VERT = 0
HORIZ_VERT = -1

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
flipped_h = cv2.flip(image, HORIZ)
rotated = imutils.rotate(flipped_h, 45)
flipped_v = cv2.flip(rotated, VERT)

print(flipped_v[189, 441])
