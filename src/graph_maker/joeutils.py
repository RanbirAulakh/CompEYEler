"""
Utilities for use with opencv
Author: Joe Bartelmo
"""
import cv2
import calendar
import time

POSITION_X = 40
POSITION_Y = 30

def flush(winname):
    """
    Repeatidly displays an image until ESC is pressed`
    """
    print('ESC to exit image')
    delay = 100
    while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) > 0: 
        k = cv2.waitKey(delay)

        # ESC pressed
        if k == 27 or k == (65536 + 27):
            action = 'exit'
            return False

def imshow(matrix, wait=False):
    """
    Displays an image, moves it to a readable location, and waits for ESC to be
    pressed
    """
    winname = str(calendar.timegm(time.gmtime()))
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, POSITION_X, POSITION_Y)
    cv2.imshow(winname, matrix)
    if wait == False:
        flush(winname)
        cv2.destroyAllWindows()
