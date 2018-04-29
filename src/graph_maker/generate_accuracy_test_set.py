"""
Generates graphs so we can see what epochs are best for the network
also lets us compare accuracy

@author Joe Bartelmo

"""
import sys
sys.path.append("../segmenter")
sys.path.append("../neural_network")
sys.path.append("../conv_segment")
from segment import segment
from conv_segment import perform_conv_segmentation
from decode import decode
#from joeutils import imshow, flush
import cv2
import numpy as np

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
            # if you're a space
            if (image_characters[row][col].astype(np.uint8) - 255).sum() == 0:
                result += " "
            else:
                import matplotlib.pyplot as plt
                plt.imshow(image_characters[row][col])
                plt.show()
                char = input("Enter Character>")
                result += char
    return result

def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python generate_accuracy_test_set.py <image>')
        print('\t<image>: Image to load from disk to test off of with segmenter')
        sys.exit()
    print(sys.argv[1])
    res = train_compeyele(cv2.imread(sys.argv[1]))
    with open("output.txt", "w") as text_file:
        text_file.write(res)

if __name__ == '__main__':
    main()
