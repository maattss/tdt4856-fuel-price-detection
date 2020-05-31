import numpy as np
import pytesseract
import argparse
import cv2

from PIL import Image

image = cv2.imread('images/two.png')
orig = image.copy()

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Color boundaries in BGR format
lower_red = [110, 90, 150] # 80, 45, 180
upper_red = [175, 180, 255] # 150, 150, 255

lower_red_hsv = [0, 65, 72]
upper_red_hsv = [9, 53, 100]

lower = np.array(lower_red, dtype=np.uint8)
upper = np.array(upper_red, dtype=np.uint8)

mask = cv2.inRange(image, lower, upper)

masked = cv2.bitwise_and(image, image, mask=mask)

# orig = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

cv2.imshow("Original and masked", np.hstack([image, masked]))
cv2.waitKey(0)

altered = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
altered = cv2.GaussianBlur(altered, (3, 3), 0)

_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
altered = cv2.erode(altered, _kernel, iterations=2)

#altered = cv2.adaptiveThreshold(altered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
*_, altered = cv2.threshold(altered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("Yo!", altered)
cv2.waitKey(0)
