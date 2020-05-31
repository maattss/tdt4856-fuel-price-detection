from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# read image from file
img = cv2.imread("images/one.png")

# pre process image by converting to greyscale and computing an edge map
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 250, 255)


# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort contours in descending order
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

output = edged.copy()


for c in cnts:
	cv2.drawContours(output, [c], -1, (240, 0, 159), 1)
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.095 * peri, True)

	if len(approx) == 4:
		displayCnt = approx
		break

x, y, w, h = cv2.boundingRect(approx)


reshaped = four_point_transform(img, displayCnt.reshape(4, 2))
height, width, channels = reshaped.shape
lineThickness = 4
# draw line on right side of image
reshaped = cv2.line(reshaped, (width, 0), (width, height), (0, 0, 0), lineThickness)
# greyscale
grey = cv2.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)


thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


# find contours in the threshold image, then initialize the digit contours list
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

print(thresh.shape)

# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour and print for debugging purposes
	(x, y, w, h) = cv2.boundingRect(c)
	#reshaped_2 = reshaped.copy()
	#v2.rectangle(reshaped_2, (x, y), (x + w, y + h), (0, 255, 0), 2)
	#cv2.imshow("conts", reshaped_2)
	#cv2.waitKey(0)
	# print("\nx: " + str(x) + "\ny: " + str(y) + "\nw: " + str(w) + "\nh: " + str(h))

	# if the contour is sufficiently large it must be a digit
	if w > 5 and (h >= 25 and h <= 30):
		digitCnts.append(c)

digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

# loop over each digit-contour
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]

	# compute the width and height off each 7 segmet to be examined
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)

	cv2.imshow("roi", roi)
	cv2.waitKey(0)

	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)), # top
		((0, 0), (dW, h // 2)),  # top-left
		((w - dW, 0), (w, h // 2)),  # top-right
		((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
		((0, h // 2), (dW, h)),  # bottom-left
		((w - dW, h // 2), (w, h)),  # bottom-right
		((0, h - dH), (w, h))  # bottom
	]
	on = [0] * len(segments)

cv2.imshow("image", reshaped)
cv2.waitKey(0)


cv2.imshow("grey", grey)
cv2.waitKey(0)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)


cv2.destroyAllWindows()

