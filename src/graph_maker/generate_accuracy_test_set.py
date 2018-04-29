"""
Generates graphs so we can see what epochs are best for the network
also lets us compare accuracy

@author Joe Bartelmo

"""
import sys
sys.path.append("../segmenter")
sys.path.append("../neural_network")
sys.path.append("../conv_segment")
from joeutils import imshow
import cv2

def train_compeyele(image, \
        conv_segment=False, \
        greedy_thresh=4.6):
    """
    Segments the image using either a greedy approach or a convolutional greedy
    approach, then converts the segmented character to text by using manual cin
    :arg image input image
    :arg conv_segment whether or not to use the convolutional segmentation
         approach
    :arg greedy_thresh value to use for binarization with greedy segmentation
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

    result = ""
    for row in range(image_characters.shape[0]):
        for col in range(image_characters.shape[1]):
            result += decode(image_characters[row][col], False)
        result += "\n"
    return result

def find_best_epoch(min_epoch=1, max_epoch=1000):
    """
    Iterates over epochs min-max, produces graphs for all in terms of accuracy
    against the training set we have
    """
    for i in range(min_epoch, max_epoch):
        train_network(min_epoch)



def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python train.py')
        sys.exit()
    find_best_epoch()

if __name__ == '__main__':
    main()
