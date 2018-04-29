"""
Generates graphs so we can see what epochs are best for the network
also lets us compare accuracy

@author Joe Bartelmo

"""
import sys
sys.path.append("..")
sys.path.append("../segmenter")
sys.path.append("../neural_network")
sys.path.append("../conv_segment")
import cv2
from neural_network.keras import train_network
from segment import segment
import pickle
import numpy as np
from helpers import crop_letter, resize_to_fit
from tensorflow.python.keras.models import load_model
import classifier

TEST_PNG = "perfect_impossible_code.png"
with open("output.txt", "r") as f:
    EXPECTED = f.read()
    EXPECTED = EXPECTED[0:len(EXPECTED)-1]

def get_accuracy(image_characters, model, lb):
    """
    Tests a model on a few defined training sets from the greedy segmenter
    :returns overallaccuracy plot, character_accuracy plot
    """
    result = ""
    for row in range(image_characters.shape[0]):
        for col in range(image_characters.shape[1]):
            image = image_characters[row][col].astype(np.uint8)
            image = crop_letter(image.astype(np.uint8))
            
            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = resize_to_fit(image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            if "_upper" in letter:
                letter = letter[0:1]
            if letter in classifier.FONT_MAP:
                letter = classifier.FONT_MAP[letter]
            result += letter
    print(len(result))
    print(len(EXPECTED))

    letter_match = {}
    total_matched = 0
    for character in range(len(EXPECTED)):
        if EXPECTED[character] not in letter_match:
            letter_match[EXPECTED[character]] = 0
        if EXPECTED[character] == result[character]:
            total_matched += 1
            letter_match[EXPECTED[character]] = letter_match[EXPECTED[character]] + 1

    return float(total_matched) / float(len(EXPECTED)), letter_match

def find_best_epoch(min_epoch=1, max_epoch=1000):
    """
    Iterates over epochs min-max, produces graphs for all in terms of accuracy
    against the training set we have
    """
    MODEL_FILENAME = "model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"
    VALIDATION_FOLDER = "segmenter_output"

    grey = cv2.imread(TEST_PNG)
    if len(grey.shape) > 2:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    image_characters = segment(grey)
    with open(MODEL_LABELS_FILENAME, 'rb') as f:
        lb = pickle.load(f)

    for i in range(min_epoch, max_epoch):
        print("TESTING EPOCH =", i)
        train_network(i, "../neural_network/images")

        # Load the trained neural network
        model = load_model(MODEL_FILENAME)

        accuracy, letter_match = get_accuracy(image_characters, model, lb)
        print("ACCURACY @EPOCH", i, ":", accuracy)
        with open('dataepoch' + str(i), 'w') as f:
            f.write(str(accuracy) + "\n")
            for key in letter_match:
                f.write(str(key) + "\t" + str(letter_match[key]) +"\n")


def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 1:
        print('usage: python find_graph.py')
        sys.exit()
    find_best_epoch()

if __name__ == '__main__':
    main()
