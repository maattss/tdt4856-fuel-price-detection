import numpy as np
import pytesseract
# import argparse
import cv2
from imutils.object_detection import non_max_suppression
from PIL import Image


default_min_conf = 0.5
default_width = 320 # 1280
default_heigt = 320 # 800
default_px = 0.05
default_py = 0.05


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
            if scoresData[x] < default_min_conf:
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


def image_preprocessing(image):
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

    altered = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    altered = cv2.GaussianBlur(altered, (3, 3), 0)

    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    altered = cv2.erode(altered, _kernel, iterations=2)

    out_image = cv2.cvtColor(altered, cv2.COLOR_GRAY2BGR)

    return out_image


def recognize_text(image_in, east_path):

    # image = cv2.imread(image_path)
    image = image_in
    orig = image.copy()

    image = image_preprocessing(image)
    # print('EYOO! ', image.shape)

    (origH, origW) = image.shape[:2]

    (newW, newH) = (default_width, default_heigt)
    rW = origW / float(newW)
    rH = origH / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Configure and load pretrained EAST detector deep neural network
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    print('Loading EAST text detector...')
    net = cv2.dnn.readNet(east_path)

    # Construct a blob and perform a forward pass on EAST net
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)

    # Decode predictions
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Loop over results
    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        dX = int((endX - startX) * default_px)
        dY = int((endY - startY) * default_py)

        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # Preprocess ROI image
        roi = cv2.resize(roi, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        roi = cv2.dilate(roi, _kernel, iterations=2)
        roi = cv2.erode(roi, _kernel, iterations=2)

        # _, roi_inv = cv2.threshold(roi, 0, 250, cv2.THRESH_BINARY_INV)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # print('Showing ROI!')
        # cv2.imshow("ROI ", roi)
        # cv2.waitKey(0)

        # apply tesseract 4 to OCR
        configs = ("-l eng --oem 1 --psm 7 load_system_dawg=false load_freq_dawg=false tessedit_char_whitelist=.0123456789 outputbase digits")
        # configs = ("--oem 1 --psm 7 tessedit_char_whitelist=.0123456789")
        text = pytesseract.image_to_string(roi, lang='eng', config=configs)

        results.append(((startX, startY, endX, endY), text))


    results = sorted(results, key=lambda r:r[0][1])

    # for ((startX, startY, endX, endY), text) in results:
	#     print("========")
	#     print("{}".format(text))

    print(f'Most most confident prediciton: {results[0]}')

    return results[0][1]


if __name__ == "__main__":

    test_image = cv2.imread('crop_boi_3.png')
    eyy = recognize_text(test_image, 'frozen_east_text_detection.pb')
