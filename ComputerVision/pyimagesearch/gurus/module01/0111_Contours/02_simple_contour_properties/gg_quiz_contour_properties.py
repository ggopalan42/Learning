# USAGE
# python contour_properties_1.py --image images/more_shapes.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find external contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()

# loop over the contours again
for (i, c) in enumerate(cnts):
	# compute the area and the perimeter of the contour
	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True)
	print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))

	# draw the contour on the image
	cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)

	# compute the center of the contour and draw the contour number
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	print("Contour #{} -- cX: {:.2f}, cY: {:.2f}".format(i + 1, cX, cY))
	cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (255, 255, 255), 4)

# show the output image
cv2.imshow("Contours", clone)
cv2.waitKey(0)

# clone the original image
clone = image.copy()

# loop over the contours
for (i, c) in enumerate(cnts):
        # fit a bounding box to the contour
        (x, y, w, h) = cv2.boundingRect(c)
        print(f'Shape: {i+1}, x = {x}, y = {y}, w = {w}, h = {h}')
        
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Bounding Boxes", clone)
cv2.waitKey(0)
clone = image.copy()

clone = image.copy()

# loop over the contours
for (i, c) in enumerate(cnts):
        # fit a minimum enclosing circle to the contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        print(f'Shape: {i+1}, radius = {radius}')
        cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)

# show the output image
cv2.imshow("Min-Enclosing Circles", clone)
cv2.waitKey(0)

