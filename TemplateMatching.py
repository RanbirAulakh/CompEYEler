# file: TemplateMatching.py
# team: compEYEler
# description: read the code images, split

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
import os
import logging

def error():
    logging.error("Usage: python3 TemplateMatching.py <test_image>")
    logging.error("<test_image> -- valid image path")
    # logging.error("<template_dataset> -- valid template dataset")
    sys.exit(0)

def main():
    # init logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s %(message)s")

    # get user's input
    if (len(sys.argv) != 2):
        error()

    # check if file exist
    test_image = sys.argv[1]
    # template_dataset = sys.argv[2]
    if (os.path.isfile(test_image) != True):
        logging.error("Invalid Image Path!")
        error()


    # read image
    img = cv2.imread(test_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original gray", img_gray)

    # threshold the image
    threshold, binarized = cv2.threshold(img_gray, 90, 190, cv2.THRESH_BINARY)
    cv2.imshow("binarized", binarized)

    # contours
    frame, contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop contours
    for c in contours:

      # get actual frame
      rect = cv2.minAreaRect(c)
      floatBox = cv2.boxPoints(rect)
      intBox = np.int0(floatBox)
      a,b,e,d = cv2.boundingRect(intBox)

      # get the letter, numbers, or whitespaces (WIP)
      cv2.imshow("get letter", cv2.resize(img[b:b+d,a:a+e], (150, 150)))

      # highlight on the picture
      contours_img = img.copy()
      cv2.drawContours(contours_img, [c], -1, (0, 0, 255), 2)
      cv2.imshow("contours", contours_img)

      input()

    cv2.waitKey(0)

#     template = cv2.imread('print hello world.png', 0)
#     template = cv2.resize(template, (0, 0), fx=0., fy=0.4)
#     w, h = template.shape[::1]  # to get the width and height of an image

#     cv2.imshow('gray', template);
#     cv2.waitKey(0)
#     cv2.imshow('gray', img_gray);
#     cv2.waitKey(0)

#     print(template.shape)
#     print(img_gray.shape)

#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#     threshold = 0.8
#     loc = np.where(res >= threshold)
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

#     cv2.imwrite("res.png", img)


main()