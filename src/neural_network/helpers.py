import cv2
import numpy as np

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]
    image = np.array(image).astype(np.uint8)

    image = cv2.resize(image, (width, height))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image.copy(), padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image.copy(), (width, height))

    # return the pre-processed image
    return image

def crop_letter(image):
    # find the contours (continuous blobs of pixels) the image
    image_inv = 255 - image
    
    # bound
    x, y, w, h = cv2.boundingRect(image_inv)
    
    if (w == 0 or h == 0):
        # this image is a space
        return image
    else:
        cut = image[y:y+h, x:x+w]
        white = 255
        image = cv2.copyMakeBorder(cut, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=white)
        return image
    
