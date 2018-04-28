# file: TemplateMatching.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
# description: read the code images, splitsc

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
import os
import logging
from sklearn.externals import joblib
from skimage.feature import hog

def error():
    logging.error("Usage: python3 TemplateMatching.py <test_image>")
    logging.error("<test_image> -- valid image path")
    sys.exit(0)

def main():
    # init logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s %(message)s")

    test_image = sys.argv[1]
    if (os.path.isfile(test_image) != True):
        logging.error("Invalid Image Path!")
        error()

    # read original image
    image = cv2.imread(test_image)

    # convert to grayscale image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original gray", img_gray)

    # threshold the image, convert it to black/white
    thresh, im_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("black and white", im_bw)

    # find contours
    im_bw_copy = im_bw.copy()
    frame, contours, hierarchy = cv2.findContours(im_bw_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    rects = [cv2.boundingRect(c) for c in contours]
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)

        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_bw[pt1:pt1+leng, pt2:pt2+leng]

        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

    cv2.imshow("Done", image)
    cv2.waitKey(0)

main()
