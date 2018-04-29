"""
CompEYEler application that translates images of code into text.

@author Joe Bartelmo
@author Ka Bo Cheung
@author Robbie Kubiniec
@author Brandon Ha
@author Ranbir Aulakh

"""
import sys
sys.path.append("segmenter")
sys.path.append("neural_network")
sys.path.append("conv_segment")
import cv2
from segmenter.segment import segment
from conv_segment.conv_segment import perform_conv_segmentation
from neural_network.decode import decode

def compeyele(image, \
        conv_segment=False, \
        greedy_thresh=4.6, \
        template_match=False):
    """
    Segments the image using either a greedy approach or a convolutional greedy
    approach, then converts the segmented character to text by using either a
    neural network (default) or template matching approach.
    :arg image input image
    :arg conv_segment whether or not to use the convolutional segmentation
         approach
    :arg greedy_thresh value to use for binarization with greedy segmentation
    :arg template_match whether or not to use template matching to identify the
         image
    :return plaintext representation of the image
    """
    # greyscale if able
    grey = image.copy()
    if len(image.shape) > 2:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    if conv_segment:
        image_characters = perform_conv_segmentation(grey)
    else:
        image_characters = segment(grey, greedy_thresh)

    if template_match:
        #template match
        pass
    else:
        result = ""
        for row in range(image_characters.shape[0]):
            for col in range(image_characters.shape[1]):
                result += decode(image_characters[row][col], False)
            result += "\n"

    return result



def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python compeyeler.py <image>')
        print('\t<image> An image that you wish to get the text from')
        sys.exit()
    image = cv2.imread(sys.argv[1])
    print(compeyele(image))

if __name__ == '__main__':
    main()
