# USAGE
# python resize.py --image florida_trip_small.png

# import the necessary packages
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
(h, w) = image.shape[:2]

# of course, calculating the ratio each and every time we want to resize
# an image is a real pain -- let's create a  function where we can specify
# our target width or height, and have it take care of the rest for us.
resized = imutils.resize(image, width=2*w, inter=cv2.INTER_CUBIC)
print(resized[367, 170])
cv2.imshow("Resized via Function", resized)
cv2.waitKey(0)

