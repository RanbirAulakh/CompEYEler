from tensorflow.python.keras.models import load_model
from helpers import resize_to_fit
from helpers import crop_letter
import numpy as np
import cv2
import pickle
import sys
import os.path	
import classifier


MODEL_FILENAME = "model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
VALIDATION_FOLDER = "segmenter_output"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)


"""
Decodes an image using the pre-trained model.
"""
def decode(image, threshold=True):
    # threshold
    if threshold:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = crop_letter(image)
    
    # Re-size the letter image to 20x20 pixels to match training data
    letter_image = resize_to_fit(image, 20, 20)

    # Turn the single image into a 4d list of images to make Keras happy
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    # Ask the neural network to make a prediction
    prediction = model.predict(letter_image)

    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(prediction)[0]
    if letter in classifier.FONT_MAP:
        return classifier.FONT_MAP[letter]
    return letter

def main():
    if len(sys.argv) < 2:
        print('Usage: python decode.py <input>')
        # print('Good threshold values are between 0 and 10, but may be bigger for larger fonts')
        sys.exit()

    img = cv2.imread(sys.argv[1], 0)

    if img is None:
        print('Invalid image path!')
        print('Usage: python decode.py <input>')
        sys.exit()

    print(decode(img))
    
    
if __name__ == "__main__":
    main()
