# file: Main.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
# description: deskewing text

import cv2
import numpy as np
import math
import logging

class UnskewImage(object):

    def unSkewTheImage(self, image, debug):

        image = cv2.imread(image)

        if debug:
            cv2.imshow("Original image", image)

        # convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_not(gray_image)

        if debug:
            cv2.imshow("grayImage", gray_image)

        logging.info("Unskewing the image...")

        height, width = image.shape[:2]

        # cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
        lines = cv2.HoughLinesP(gray_image, 1, np.pi / 180, 80, width // 2, 20)
        logging.debug("Lines Detected: {0}".format(lines))
        logging.debug("Total Lines: {0}".format(len(lines)))

        angle = 0

        image_lines = image.copy()
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
                angle += math.atan2(y2 - y1, x2 - x1)

        logging.debug("Angle: {0}".format(angle))
        angle = (angle / len(lines)) * 180 / np.pi
        logging.debug("After Angle: {0}".format(angle))

        if debug:
            cv2.imshow("Display Lines on Images", image_lines)

        center = (width // 2, height // 2)
        rotated_image = cv2.getRotationMatrix2D(center, angle, 1.0)
        final_image = cv2.warpAffine(image, rotated_image, (width, height), flags=cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_REPLICATE)

        if debug:
            cv2.imshow("Rotated Image", final_image)

        logging.info("Fixed text rotation! The angle is {0}.".format(angle))

        if debug:
            cv2.waitKey(0)

    # http://felix.abecassis.me/2011/09/opencv-detect-skew-angle/

