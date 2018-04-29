"""
Generates graphs so we can see what epochs are best for the network
also lets us compare accuracy

@author Joe Bartelmo

"""
import sys
sys.path.append("../segmenter")
sys.path.append("../neural_network")
sys.path.append("../conv_segment")
import cv2
from keras import train_network

TEST_PNG = "perfect_impossible_code.png"
EXPECTED = "output.txt"

def get_accuracy(image_characters, model):
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
    print(result)

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
    if len(image.shape) > 2:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    if conv_segment:
        image_characters = perform_conv_segmentation(grey)
    else:
        image_characters = segment(grey, greedy_thresh)

    for i in range(min_epoch, max_epoch):
        print("TESTING EPOCH =", i)
        train_network(min_epoch)

        MODEL_FILENAME = "model.hdf5"
        MODEL_LABELS_FILENAME = "model_labels.dat"
        VALIDATION_FOLDER = "segmenter_output"

        with open(MODEL_LABELS_FILENAME, "rb") as f:
            lb = pickle.load(f)

        # Load the trained neural network
        model = load_model(MODEL_FILENAME)

        accuracy, letter_match = get_accuracy(image_characters, model)
        print("ACCURACY @EPOCH", i, ":", accuracy)



def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python find_graph.py')
        sys.exit()
    find_best_epoch()

if __name__ == '__main__':
    main()
