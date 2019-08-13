# USAGE
# python tictactoe.py

# import the necessary packages
import cv2
import imutils

# load the tic-tac-toe image and convert it to grayscale
image = cv2.imread("more_shapes_example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find all contours on the tic-tac-toe board
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for (i, c) in enumerate(cnts):
	# compute the area of the contour along with the bounding box
	# to compute the aspect ratio
	area = cv2.contourArea(c)
	(x, y, w, h) = cv2.boundingRect(c)
	aspect_ratio = '{:0.2f}'.format(w/float(h))
	hull = cv2.convexHull(c)
	hullArea = cv2.contourArea(hull)
	solidity = area / float(hullArea)
	extent = area / float(w * h)


	print('Aspect Ratio = {}'.format(aspect_ratio))
	print('Solidity = {}'.format(solidity))
	print('Extent = {}'.format(extent))
	cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
	cv2.putText(image, aspect_ratio, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

    


# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
