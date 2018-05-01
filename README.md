# CompEYEler

## Usage
```
# note, all require python 3

# main application
python compeyeler.py <image>

# template matching 
python template_matching.py -i <character image>

# train neural network
python keras.py

# test decoder on neural network
python decoder.py <character image>

# convolutional segmenter
python conv_segment.py <image>

# segmenter
python segment.py <image>

# generate epoch data from (train neural networks and test against a black box image that isn't in training set)
python find_graphs.py
# to be run after find_graphs.py (seperated because my computer crashed in the middle of find_graphs once)
# generates the matplotlib graphs for the epoch data
python make_graphs.py


# generate training images from TTF on your operating system
python ttf2png.py <folder of where to search for ttf>

```
_<image> supplied is the location of the image on your OS_

## Summary 

Analyzes code images of different fonts and different capture environments and converts them to text.

### Objective

The objective of our project is to detect and parse text from images and write the result into text files. The minimum viable product is to be able to interpret mono-spaced, whitespace sensitive images of text into text that a user can compile into a working program.

### Introduction

Have you ever wanted to copy code from an image? Want to collaborate more efficiently? Want to know what kind of language a snippet is in? Look no further! CompEYEler is a computer vision tool that converts code images to text, without you having to transcribe the entire code yourself. CompEYEler will change the way you share code with friends, peers, teammates, and more. 

