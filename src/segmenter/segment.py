import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import sys
import os.path
from os import walk
import scipy.stats
import scipy.signal

"""
Using a crossings array and a threshold, determine the
width and offset needed to divide the image into fixed width segments.

Inputs:
crossings - a 1D array of how many times the color swapped for the given column of data
threshold - the first result that is below this threshold will be returned,
    set higher for blurrier images.

Outputs:
repeater - float describing a width of a cell
offset - integer describing when the first cell starts
"""
def accumulate(crossings, threshold, search_start, search_space_multiplier):
    size = crossings.shape[0]
    best_offset = 0
    best_repeater = 2
    best_accumulator = float('inf')
    possible_repeaters = np.linspace(search_start, search_start * search_space_multiplier, num=500)
    for true_repeater in possible_repeaters:
        for offset in range(search_start):
            accumulator = 0
            iterations = int((size - offset) / true_repeater)
            for iteration in range(iterations):
                accumulator += crossings[int(true_repeater * iteration) + offset]
            # normalize
            accumulator /= iterations
            # print(accumulator)
            if accumulator < best_accumulator:
                best_offset = offset
                best_repeater = true_repeater
                best_accumulator = accumulator
                # print(str(best_offset) + "\t" + str(best_repeater) + "\t" + str(best_accumulator))
                # if threshold > 0:
                #     if best_accumulator < threshold:
                #         return best_repeater, int(best_offset % best_repeater)

    # print(best_accumulator)
    # print(best_offset)
    # print(best_repeater)

    return best_repeater, int(best_offset % best_repeater)

"""
Determines the number of crossings for each row of data in the image.
Image must be binary.

Input:
    img - binary image
    axis -  0 counts crossings per row (for each y)
            1 counts crossings per column (for each x)

Output:
    crossing_counts - 1D array
"""
def crossings(img, axis=0):
    if axis == 1:
        img = np.transpose(img)

    size = img.shape[0]

    crossing_counts = np.zeros((size))

    for col in range(size):
        crossings = 0
        crossing_state = img[col][0]
        for row in img[col]:
            if row != crossing_state:
                crossing_state = row
                crossings += 1
        crossing_counts[col] = crossings

    # print(crossing_counts)
    return crossing_counts

"""
Returns a image thresholded with otsu's method. Inverts the image if it is mostly dark.
"""
def standardize(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    avg = np.average(img)
    if avg < 128:
        img = 255 - img

    return img

"""
Returns binary image divided into equal sized chunks.
"""
def segment(img, threshold=4.6):
    print("Segmenter started")  
    
    img2 = standardize(img)

    edges = cv2.Canny(img2, 100, 200)

    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = 0
    max_height = 0
    widths = []
    heights = []
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] > max_width:
            max_height = rect[2]
        if rect[3] > max_height:
            max_width = rect[3]
        widths.append(rect[2])
        heights.append(rect[3])
        x,y,w,h = rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    widths = sorted(widths)
    heights = sorted(heights)

#     print(heights)

    ninety_five_width = widths[len(widths)//20 * 19]
    ninety_five_height = heights[len(heights)//20 * 19]

    # cv2.imshow("Show", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # print(ninety_five_width)
    # print(ninety_five_height)

    img = img2

    x_crossing_counts = crossings(img, axis=1)
    x_repeater, x_offset = accumulate(x_crossing_counts, threshold, ninety_five_width, 2)

    y_crossing_counts = crossings(img, axis=0)
    
    # we need the multiplier to be higher, since there's more space vertically between letters
    y_repeater, y_offset = accumulate(y_crossing_counts, threshold, ninety_five_height, 3)

    x_iterations = int((img.shape[1] - x_offset) / x_repeater) + 1
    y_iterations = int((img.shape[0] - y_offset) / y_repeater) + 1

    print("Block width: " + str(x_repeater))
    print("Block height: " + str(y_repeater))
    print("Optimal x-offset: " + str(x_offset))
    print("Optimal y-offset: " + str(y_offset))
    print("Segmenter ended")
    print("")

    segments = np.full((y_iterations, x_iterations, int(y_repeater + 1), int(x_repeater + 1)), fill_value = 255)
    for y_iter in range(y_iterations):
        for x_iter in range(x_iterations):
            top_bound = int(y_iter * y_repeater) + y_offset
            bottom_bound = int((y_iter + 1) * y_repeater) + y_offset
            left_bound = int(x_iter * x_repeater) + x_offset
            right_bound = int((x_iter + 1) * x_repeater) + x_offset
            letter = img[top_bound:bottom_bound, left_bound:right_bound]

            segments[y_iter][x_iter][:letter.shape[0], :letter.shape[1]] = letter

    return segments

"""
Run sample code on the input.
"""
def main():
    if len(sys.argv) < 2:
        print('Usage: python segment.py <input>')
        # print('Good threshold values are between 0 and 10, but may be bigger for larger fonts')
        sys.exit()

    img = cv2.imread(sys.argv[1], 0)

    if img is None:
        print('Invalid image path!')
        print('Usage: python segment.py <input>')
        sys.exit()

    # threshold = float(sys.argv[2])

    # if threshold < 0:
    #     print('Threshold must be 0 or more!')
    #     print('Usage: python segment.py <input>')
    #     # print('Good threshold values are between 0 and 10, but may be bigger for larger fonts')
    #     sys.exit()

    # increase threshold for debugging
    np.set_printoptions(threshold=10000000)

    segments = segment(img, 100)
    # print(segments)

    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            cv2.imwrite('./segmenter_output/' + str(row) + "-" + str(col) + ".png" , segments[row][col])
    print('segments written to /segmenter_output')

if __name__ == "__main__":
    main()
