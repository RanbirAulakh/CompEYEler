# file: DistortImage.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
# description: return 3 distort images...

import cv2
import numpy as np
import math
import logging
import random

class DistortImage(object):

    def waveImage(image):
        distort_2_output = np.zeros(image.shape, dtype=image.dtype)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
                offset_y = 0
                if j + offset_x < image.shape[0]:
                    distort_2_output[i, j] = image[i, (j + offset_x) % image.shape[1]]
                else:
                    distort_2_output[i, j] = 0

        return distort_2_output

    def blurImage(image):
        blur = cv2.GaussianBlur(image, (5,5), 17)

        return blur

    def rainbowImage(image):
        rainbow = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)

        # cv2.imshow("Distort 3 (COLORMAP_SUMMER)", cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW))
        # cv2.imshow("Distort 3 (COLORMAP_JET)", cv2.applyColorMap(image, cv2.COLORMAP_JET))

        return rainbow

    def tiltImage(image):
        center = (image.shape[0] // 2, image.shape[1] // 2)
        angle = random.uniform(-360, 360)

        rotated_image = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotated_image, (image.shape[0], image.shape[1]), flags=cv2.INTER_AREA,
                                         borderMode=cv2.BORDER_REPLICATE)

        return rotated_image

    def randTiltImage(image):
        image_list = []
        for i in range(0, 5):
            center = (image.shape[0] // 2, image.shape[1] // 2)
            angle = random.uniform(-360, 360)

            rotated_image = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotated_image, (image.shape[0], image.shape[1]), flags=cv2.INTER_AREA,
                                         borderMode=cv2.BORDER_REPLICATE)
            image_list.append(rotated_image)

        return image_list

    def thresholdImage(image):
        blur_image = cv2.GaussianBlur(image, (5,5), 11)
        thres, binarized = cv2.threshold(blur_image, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return binarized

    def upsideDown(image):
        center = (image.shape[1] // 2, image.shape[0] // 2)

        upside_down_image = cv2.getRotationMatrix2D(center, 180, 1.0)
        upside_down_image = cv2.warpAffine(image, upside_down_image, (image.shape[1], image.shape[0]), flags=cv2.INTER_AREA,
                                         borderMode=cv2.BORDER_REPLICATE)

        return upside_down_image

    def DistortImage(self, image, debug):
        distort_outputs = []
        image = cv2.imread(image)

        if debug:
            cv2.imshow("Original image", image)

        # convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_not(gray_image)


        if debug:
            cv2.imshow("grayImage", gray_image)

        logging.info("Distorting the image...")

        distort1_Image = DistortImage.blurImage(image)
        distort_outputs.append(distort1_Image)
        if debug:
            cv2.imshow("Distort 1 (Blur)", distort1_Image)

        distort2_Image = DistortImage.waveImage(image)
        distort_outputs.append(distort2_Image)
        if debug:
            cv2.imshow("Distort 2 (Wave)", distort2_Image)

        distort3_Image = DistortImage.rainbowImage(image)
        distort_outputs.append(distort3_Image)
        if debug:
            cv2.imshow("Distort 3 (Rainbow)", distort3_Image)

        distort4_Image = DistortImage.randTiltImage(image)
        for i in range(len(distort4_Image)):
            distort_outputs.append(distort4_Image[i])
            if debug:
                cv2.imshow("Distort " + str( (i + 4) )  + " (Tilted)", distort4_Image[i])

        distort8_Image = DistortImage.tiltImage(distort1_Image)
        distort_outputs.append(distort8_Image)
        if debug:
            cv2.imshow("Distort 8 (Tilted + Blur)", distort8_Image)

        distort9_Image = DistortImage.thresholdImage(gray_image)
        distort_outputs.append(distort9_Image)
        if debug:
            cv2.imshow("Distort 9 (Threshold)", distort9_Image)

        distort10_Image = DistortImage.upsideDown(image)
        distort_outputs.append(distort10_Image)
        if debug:
            cv2.imshow("Distort 10 (Upside Down)", distort10_Image)

        if debug:
            cv2.waitKey(0)

        return distort_outputs

