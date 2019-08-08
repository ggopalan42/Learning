# USAGE
# python mag_orientation.py --image coins02.png

# import the necessary packages
import numpy as np
import cv2

gX = -7
gY = 187
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

print(f'Orientation: {orientation}')


M = np.array([[44, 67, 96], [231, 184, 224], [51, 253, 36]])
gY_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gX_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

sobel_gY = np.sum(np.multiply(M, gY_kernel))
sobel_gX = np.sum(np.multiply(M, gX_kernel))


print(f'Y-Sobel: {sobel_gY}')
print(f'X-Sobel: {sobel_gX}')

sob_or = np.arctan2(sobel_gY, sobel_gX) * (180 / np.pi) % 180
print(f'Sobel Ori: {sob_or}')
