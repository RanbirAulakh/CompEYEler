#!/usr/bin/env python
import sys
import os
import ntpath
from fontTools.ttLib import TTFont
from subprocess import Popen, PIPE

IMAGES_DIR = "images"
import glob


def main(folder):
    """
    Uses imagemagick to extract fonts to images
    """
    files = glob.glob(folder + '/**/*.ttf')

    for TTF_PATH in files:
        FONT_SIZES = [18]
        TTF_NAME, TTF_EXT = os.path.splitext(os.path.basename(TTF_PATH))

        ttf = TTFont(TTF_PATH, 0, allowVID=0, \
                ignoreDecompileErrors=True, fontNumber=-1)

        basename = ntpath.basename(TTF_PATH)

        if not os.path.isdir(basename):
            os.mkdir(basename)
            os.mkdir(basename + "/18pt")

        characters = []
        shortcut = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for x in ttf["cmap"].tables:
            for y in x.cmap.items():
                char_unicode = chr(y[0])
                if char_unicode in shortcut:
                    characters.append((y[1], char_unicode))
        ttf.close()

        for FONT_SIZE in FONT_SIZES:
            for char_tuple in characters:
                char_name, character = char_tuple
                imagemagick = ["convert", \
                        "-background", "white", \
                        "-fill", "black",  \
                        "-font", TTF_PATH, \
                        "-pointsize", str(FONT_SIZE), \
                        "label:\\" + character+"", \
                        basename + "/" + str(FONT_SIZE) + "pt/" + char_name +"_upper.png"]
                line = " ".join(imagemagick)
                print(line)
                process = Popen(line, stderr = PIPE, stdout = PIPE, shell = True)
                (output, err) = process.communicate()
                exit_code = process.wait()

if __name__ == '__main__':
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python ttf2png.py <folder>')
        print('\t<folder> location with which to extract the true type fonts')
        print('\t\tttf folder on ubuntu: /usr/share/fonts/truetype') 
        sys.exit()
    
    main(sys.argv[1])

