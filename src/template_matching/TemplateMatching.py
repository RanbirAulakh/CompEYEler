# file: TemplateMatching.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
# description: read the code images, splitsc

import cv2
import numpy as np
import os
import logging
import argparse
import helpers
import sys
sys.path.append("../neural_network/")

def template_segment(image):
    # convert to gray scale image
    character_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original gray", character_image)

    # threshold the image, convert it to black/white
    thresh, im_bw = cv2.threshold(character_image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = cv2.bitwise_not(im_bw)

    frame, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]

    # https://github.com/RanbirAulakh/CompEYEler/commit/fab90b2cf8290c939191242d0403b1bd24f36349
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)

    leng = int(rect[3] * 1.6)

    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

    roi = im_bw[pt1:pt1+leng, pt2:pt2+leng]

    roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3,3))

    return roi

def template_match_char(char_image, threshold):
    # read character image
    image = cv2.imread(char_image)

    # convert to gray scale image
    character_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original gray", character_image)

    # threshold the image, convert it to black/white
    thresh, im_bw = cv2.threshold(character_image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = cv2.bitwise_not(im_bw)

    # resize to fit then crop
    crop_image = helpers.crop_letter(im_bw)
    cv2.imshow("Cropped Letter", crop_image)

    resized_image = helpers.resize_to_fit(crop_image.astype(np.uint8), 20, 20)
    cv2.imshow("Resized Letter", resized_image)

    # directory full of texts -- to check if it's match
    text_directory = "../../Images/texts/"
    if os.path.isdir(text_directory) is not True:
        logging.warning("It is not a directory!")
        sys.exit(0)

    # reads templates
    text_files = os.listdir(text_directory)
    for i in text_files:
        logging.debug("Matching " + i)
        template_image = cv2.imread(text_directory + i)
        template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # convert to grayscale and to black and white
        _, template_bw = cv2.threshold(template_image_gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        template_bw = cv2.bitwise_not(template_bw)
        # cv2.imshow("template_bw", template_bw)

        # resize to fit then crop
        tm_crop_image = helpers.crop_letter(template_bw)
        # cv2.imshow("tm_Cropped Letter", tm_crop_image)
        tm_resized_image = helpers.resize_to_fit(tm_crop_image.astype(np.uint8), 20, 20)
        # cv2.imshow("tm_Resized Letter", tm_resized_image)

        result = cv2.matchTemplate(resized_image, tm_resized_image, cv2.TM_CCOEFF_NORMED)
        (minValue, maxValue, minLocation, maxLocation) = cv2.minMaxLoc(result)
        # (x, y) = minLocation
        # cv2.rectangle(character_image, (x, y), (x + template_image.shape[0], y + template_image.shape[1]), (0, 255, 0), 2)
        # cv2.imshow("Source", character_image)

        logging.debug("Result Accuracy: ", (minValue, maxValue, i))

    cv2.waitKey(0)


def main():
    # init logging and argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                        help="Image Input Path",
                        required=True)
    parser.add_argument("-t", "--threshold",
                        help="Threshold Value",
                        required=False)
    parser.add_argument("-d", "--debug",
                        help="Enable Debug",
                        required=False, action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s - %(levelname)s %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s - %(levelname)s %(message)s", level=logging.INFO)

    image = args.image
    debug = args.debug
    threshold = args.threshold
    if threshold == "" or threshold == None:
        threshold = 100

    template_match_char(image, threshold)


main()
