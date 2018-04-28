# file: Main.py
# team: compEYEler
# authors: Ranbir Aulakh, Joseph Bartelmo,
#           Ka Bo Cheung, Brandon Ha, Robbie Kubiniec
#

import argparse
import logging
from UnskewImage import UnskewImage
from DistortImage import DistortImage

import itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                      help="Image Input Path",
                      required=True)
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

    # unskew the image
    #unskewImage = UnskewImage()
    #unskewImage.unSkewTheImage(image, debug)

    # distort the image
    distortImage = DistortImage()
    distortImage.DistortImage(image, debug)


main()