import numpy as np
import pytesseract
import argparse
import cv2
from imutils.object_detection import non_max_suppression
from PIL import Image


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

default_width = 1280
default_heigt = 800


def recognize_text(image_path, east_path):

    image = cv2.imread(image_path)
    orig = image.copy()

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

    print('[INFO] Loading EAST text detector...')
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

        dX = int((endX - startX) * args["padding_x"])
        dY = int((endY - startY) * args["padding_y"])

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
