import sys
from segment import segment
import cv2
import os.path
sys.path.append("..")
import neural_network.classifier


inv_map = {v: k for k, v in neural_network.classifier.FONT_MAP.items()}

def main():
    if len(sys.argv) < 4:
        print('Usage: python segment.py <image> <plaintext> <output folder>')
        sys.exit()

    img = cv2.imread(sys.argv[1], 0)

    if img is None:
        print('Invalid image path!')
        print('Usage: python segment.py <input>')
        sys.exit()

    segments = segment(img, 100)
    # print(segments)

    counts = {}

    with open(sys.argv[2]) as f:
        for row, line in enumerate(f):
            for col, letter in enumerate(line):
                if letter == '\n' or letter == '\r':
                    continue
                label = letter
                # print(str(row) + " " + str(col))
                if letter in inv_map:
                    label = inv_map[letter]
                # print(label)    
                
                save_path = sys.argv[3]
                
                # make folder
                if not os.path.exists(save_path):	
                    os.makedirs(save_path)
                
                save_path = os.path.join(save_path, label)
                
                # make folder
                if not os.path.exists(save_path):	
                    os.makedirs(save_path)
                
                count = counts.get(label, 1)
                
                if count > 20:
                    # we have too much of the character, skip it
                    continue
                
                try:
                    save_path = os.path.join(save_path, label + (".png" * count))	
                    # print(save_path)
                    cv2.imwrite(save_path, segments[row][col])
                except:
                    pass
                
                counts[label] = count + 1
                

if __name__ == "__main__":
    main()
