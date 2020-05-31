# Text recognition in Python

# Import neccesary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

from PIL import Image

print('Succesful imports')

# Decode predictions from the EAST detector
def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < args['min_confidence']:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use geometry volume to derive the width and height of bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            startX = int(endX - w)

            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startY = int(endY - h)

            # Append the bounding box coordinates and probability score
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# Construct argparser
ap = argparse.ArgumentParser()
ap.add_argument('-i', "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, # default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", '--width', type=int, default=1280,
	help="nearest multiple of 32 for resized width")
#ap.add_argument("-w", '--width', type=int, default=320,
#	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=800,
	help="nearest multiple of 32 for resized height")
#ap.add_argument("-e", "--height", type=int, default=320,
#	help="nearest multiple of 32 for resized height")
ap.add_argument("-p_x", "--padding_x", type=float, default=0.05,
	help="amount of padding to add to x border of ROI")
ap.add_argument("-p_y", "--padding_y", type=float, default=0.05,
    help="amount of padding to add to y border of ROI")
args = vars(ap.parse_args())

# Load and preprocess image
image = cv2.imread(args['image'])
orig = image.copy()

(origH, origW) = image.shape[:2]

(newW, newH) = (args['width'], args['height'])
rW = origW / float(newW)
rH = origH / float(newH)

# Resize image and grab new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Configure pretrained east detector deep neural network
layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

print('[INFO] Loading EAST text detector...')
net = cv2.dnn.readNet(args['east'])

# Construct a blob and perform a forward pass
blob = cv2.dnn.blobFromImage(
    image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
)
net.setInput(blob)
(scores, geometry) = net.forward(layer_names)

# Decode predictions
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# Looping over bounding boxes and processing results
results = []
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    dX = int((endX - startX) * args["padding_x"])
    dY = int((endY - startY) * args["padding_y"])

    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]

    # Rescale roi to 300 dpi
    (roiH, roiW) = roi.shape[:2]
    _factor = min(1, float(1024.0 / roiH))
    new_roiH = int(_factor * roiH)
    new_roiW = int(_factor * roiW)

    roi_resized = cv2.resize(roi, (new_roiW, new_roiH))

    #roi = roi_resized   #################!!!!!!!!!!!!!!!!!!!####################

    # Preprocess ROI image
    roi = cv2.resize(roi, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    #roi = cv2.UMat(roi)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    #roi = cv2.GaussianBlur(roi, (5, 5), sigmaX=5)
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #_kernel = np.ones((12,12), np.uint8)
    #roi = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, _kernel)
    #_, roi = cv2.threshold(roi, 0, 250, cv2.THRESH_OTSU)
    #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #roi = cv2.UMat(roi)
    roi = cv2.dilate(roi, _kernel, iterations=2)
    roi = cv2.erode(roi, _kernel, iterations=2)
    #print(type(roi))
    #_, roi_inv = cv2.threshold(roi, 0, 250, cv2.THRESH_BINARY_INV)
        #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, roi_inv = cv2.threshold(roi, 0, 250, cv2.THRESH_BINARY_INV)

    #roi = roi_inv
    #print(type(roi_inv))

    #kernel = np.ones((1, 1), np.uint8)
    #roi = cv2.dilate(roi, kernel, iterations=1)
    #roi = cv2.erode(roi, kernel, iterations=1)

    # roi = cv2.bilateralFilter(roi, 9, 75, 75)
    # roi = cv2.medianBlur(roi, 3)
    #roi = cv2.medianBlur(roi, 7)

    #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #roi = cv2.adaptiveThreshold(roi, 155, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    *_, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    print('Showing!')
    cv2.imshow("ROI ", roi)
    cv2.waitKey(0)

    # apply tesseract 4 to OCR
    configs = ("-l eng --oem 1 --psm 7 load_system_dawg=false load_freq_dawg=false tessedit_char_whitelist=.0123456789 outputbase digits")
    #configs = ("--oem 1 --psm 7 tessedit_char_whitelist=.0123456789")
    text = pytesseract.image_to_string(roi, lang='eng', config=configs)

    results.append(((startX, startY, endX, endY), text))

# Display results to verify working
results = sorted(results, key=lambda r:r[0][1])

if not results:
    print('No results!')

for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}".format(text))
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)
