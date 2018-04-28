# file: UnskewImage.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
# description: deskewing text

import cv2
import numpy as np
import math
import logging
from matplotlib import pyplot as plt

class UnskewImage(object):

    def unSkewTheImage(self, image, debug):

        logging.info("Unskewing the image...")

        image = cv2.imread(image)

        if debug:
            cv2.imshow("Original image", image)

        # convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, binarized = cv2.threshold(gray_image, 225, 255, cv2.THRESH_BINARY)
        binarized_inv = cv2.bitwise_not(binarized)


        if debug:
            cv2.imshow("binarized", binarized)
            cv2.imshow("binarized_inv", binarized_inv)

        elements = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        print(elements)
        erode_image = cv2.erode(binarized_inv, elements)

        if debug:
            cv2.imshow("erode_image", erode_image)

        height, width = erode_image.shape[:2]
        points = []

        for i in range(height):
            for j in range(width):
                if erode_image[i][j]:
                    points.append( [i,j] )

        print(np.zeros(erode_image.shape))
        points = np.array(points)

        box = cv2.minAreaRect(points)
        angle = (box[2])
        if angle < -45:
            angle += 90
        print("angle", angle)

        vertices = cv2.boxPoints(box)

        final_image = erode_image.copy()
        for i in range(len(vertices)):
            final_image = cv2.line(erode_image, (vertices[i][0], vertices[i][1]),
                                   (vertices[(i + 1) % 4][0], vertices[(i + 1) % 4][1]), (255, 0, 0), 1, cv2.LINE_AA)

        if debug:
            cv2.imshow("final_image", final_image)

        cv2.waitKey(0)
