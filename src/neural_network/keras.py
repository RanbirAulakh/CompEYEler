# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:09:50 2018

@author: Kami
"""

import cv2	
import pickle	
import os.path	
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras._impl.keras.layers.core import Flatten, Dense
from helpers import resize_to_fit
from helpers import crop_letter

def train_network(num_epochs=30, LETTER_IMAGES_FOLDER="images"):
    """
    Trains a convolutional neural network to our training set and outputs the 
    model to a local file to be read and used
    """
    MODEL_FILENAME = "model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"


    # initialize the data and labels
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file, 0)
        
        if image is None:
            # the filename was probably too long
            print("WARNING: We couldn't open this image, its filename is probably too long:")
            print(image_file)
            continue
        
        
        # threshold
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = crop_letter(image)
        
        # Resize the letter so it fits in a 20x20 pixel box
        image = resize_to_fit(image, 20, 20)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-1].replace('.png', '')

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 70 nodes (one for each possible letter/number we predict)
    model.add(Dense(97, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=num_epochs, verbose=1)

    # Save the trained model to disk
    model.save(MODEL_FILENAME)

if __name__ == '__main__':
    train_network();
