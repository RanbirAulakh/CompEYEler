"""
Utilities for use with opencv
Author: Joe Bartelmo
"""
import cv2
import calendar
import time

POSITION_X = 400
POSITION_Y = 300

def flush():
    """
    Repeatidly displays an image until ESC is pressed`
    """
    cv2.destroyAllWindows()
    cv2.waitKey(10)

def imshow(matrix):
    """
    Displays an image, moves it to a readable location, and waits for ESC to be
    pressed
    """
    winname = str(calendar.timegm(time.gmtime()))
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, POSITION_X, POSITION_Y)
    cv2.imshow(winname, matrix)
    cv2.waitKey(1)
