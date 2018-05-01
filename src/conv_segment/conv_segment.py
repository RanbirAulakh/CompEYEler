"""
Convolutional approach to the segmenter

@author Joe Bartelmo
@author Robbie Kubiniec

"""
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from scipy.stats import mode

def compute_avg_character_width(src, space=False):
    """
    Takes in 1 binary row and extracts data from it to reveal the width of each
    character then computes the data
    :src input binary row
    :output (avg_length, mode_length, character_lengths)
    """
    current = 0
    values = []
    for col in range(src.shape[1]):
        val = src[0,col]
        if (not space and val != 0) or (space and val == 0):
            current += 1
        elif current != 0:
            values.append(current)
            current = 0
    values = np.array(values)
    return (np.mean(values), mode(values), values)

def block_me(src, vertical=True):
    """
    Convert the characters into little blocks for us to play with
    :src a binary image
    :return the first row in the image, we dont need all of them
    """
    if vertical:
        return convolve(src, np.ones((src.shape[0]*2, 1)))[0].reshape((1, src.shape[1]))
    return convolve(src, np.ones((1, src.shape[1]*2)))[:,0].reshape((src.shape[0], 1))

def segment_lines(src):
    """
    Takes in an arbitrary image and computes the lines for each image, spits
    out an array
    """
    line_locs = block_me(src, False)
    line_locs[line_locs > 0] = 255
    lines = []

    current = 0
    for row in range(line_locs.shape[0]):
        val = line_locs[row, 0]
        if val != 0:
            current += 1
        elif current != 0:
            lines.append(src[row-current:row, :])
            current = 0

    return lines

def segment_characters(line):
    """
    Takes in an arbitrary line and segments out the characters and spaces
    """
    char_locs = block_me(line, True)
    # compute the character width
    mean, mode, values = compute_avg_character_width(line)
    #print("Character Mean:", mean)
    #print("Character Mode:", mode)
    #print("Character Set:", values)
    smean, smode, svalues = compute_avg_character_width(line, True)
    #print("Space Mean:", smean)
    #print("Space Mode:", smode)
    #print("Space Set:", svalues)
    character_size = mode[0][0] + smode[0][0]
    #print("Character width = Character Mode + Space Mode =", mode[0]+smode[0])

    print(character_size)
    values = []
    offset = -1
    endoffset = -1
    for col in range(char_locs.shape[1]):
        if char_locs[0, col] != 0:
            offset = col
            break
    for col in reversed(range(char_locs.shape[1])):
        if char_locs[0, col] != 0:
            endoffset = col
            break

    index = offset
    while index < endoffset:
        values.append(line[:, index -1:index+character_size])
        index += character_size
    return values

def force_black_background(src):
    """
    Prints the homework 3 output as desired by professor
    :arg filename name of the file loaded
    :arg dice number of dice
    :arg dice_numbers map of dice numbers
    """
    grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret2, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    th = th.astype(float)
    # if three out of four corners are 255 invert the color
    i = (th[0,0] + th[0,th.shape[1] - 1] + th[th.shape[0] - 1, 0] + \
            th[th.shape[0] - 1, th.shape[1] - 1])# / 255.0
    if i/255.0 >= 3:
        th = cv2.bitwise_not(th.astype(np.uint8))
		
    return th.astype(np.uint8)

def perform_conv_segmentation(src):
    """
    Performs our convolutional segmentation algorithm to get all characters in a
    set including spaces
    """
    thresh = force_black_background(src)
    i = 0
    characters = []
    for line in segment_lines(thresh):
        chars = segment_characters(line)
        characters.append(chars)
    return np.array(characters)

def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2 and (len(sys.argv) != 3 or sys.argv[2] != '-d'):
        print('usage: python ransac.py <image>')
        print('\t<image> A single line of code as an image')
        sys.exit()
    image = cv2.imread(sys.argv[1])
    chars = perform_conv_segmentation(image)
    #print(chars)

if __name__ == '__main__':
    main()
