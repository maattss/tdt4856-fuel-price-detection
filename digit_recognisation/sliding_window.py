from helpers import pyramid
from helpers import sliding_window
import argparse
import time
import cv2


# load image and define the window width and height
image = cv2.imread("images/one.png")
(winW, winH) = (128, 48)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = resized.copy()
        # draw rectangle from (x,y) to (x+winW, y+winH)
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        clone = clone[y:(y + winH), x:(x+winW)]
        cv2.imshow("Window", clone)
        cv2.waitKey(0)
        time.sleep(0.025)
    cv2.destroyAllWindows()