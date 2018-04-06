import cv2
import glob
import struct
from ntpath import basename

"""
Location of the true type fonts ports
"""
IMAGES_DIR = "./images"

"""
Fixed Image size
"""
IMAGE_SIZE = (20,20)

"""
For keeping track of all images that do not have a name
relating to the actual character
"""
FONT_MAP = {
    "ampersand" : "&",
    "asciicircum" : "^",
    "asciitilde" : "~",
    "at" : "@",
    "bar" : "|",
    "braceleft" : "{",
    "braceright" : "}",
    "bracketleft" : "[",
    "bracketright" : "]",
    "colon" : ":",
    "comma" : ",",
    "dollar" : "$",
    "equal" : "=",
    "exclam" : "!",
    "one" : "1",
    "two" : "2",
    "three" : "3",
    "four" : "4",
    "five" : "5",
    "six" : "6",
    "seven" : "7",
    "eight" : "8",
    "nine" : "9",
    "zero" : "0",
    "grave" : "`",
    "greater" : ">",
    "less" : "<",
    "hyphen": "-",
    "numbersign": "#",
    "parenleft" : "(",
    "parenright" : ")",
    "percent": "%",
    "period" : ".",
    "plus" : "+",
    "question" : "?",
    "quotedbl" : "\"",
    "semicolon" : ";",
    "slash" : "/",
    "underscore": "_",
    "quotesingle": "'",
    "backslash" : "\\",
    "asterisk" : "*"
}

def band_num(image):
    """
    Returns the total number of bands in an image
    :arg image image to determine the band dimension of
    :return integer >= 1
    """
    if len(image.shape) == 3:
        return image.shape[2]
    return 1

def transform_image(img):
    """
    Performs a series of different operations across the image in an
    attempt to broaden the neuralnetwork training data, returns a
    list of numpy arrays of the transformed character
    :return: list of changed numpy arrays and the original numpy array
    """
    transformed = [img]
    #TODO: Transform the image in ways we would expect to see, append the
    #      different images to transformed, don't resize the image
    return transformed

def get_data():
    """
    Goes through the TTF ported images and sorts them into a list of numpy
    arrays and their corresponding labels. Resizes all numpy arrays to a
    fixed height and width and greyscales them
    labels
    :return: tuple, numpy arrays, labels
    """
    images = []
    characters = []
    for picture in glob.glob(IMAGES_DIR + '/**/18pt/*.png'):
        character = basename(picture).replace(".png", "")
        if character in FONT_MAP:
            character = FONT_MAP[character]
        if len(character) > 1:
            continue
        image = cv2.imread(picture)
        if band_num(image) != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, IMAGE_SIZE)

        different_character_images = transform_image(image)
        images += different_character_images
        characters += [ord(character)] * len(different_character_images)
    return images, characters