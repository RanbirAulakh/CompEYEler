from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def cnn_model_fn(num_classes):
    """
    Generates a model in keras for us to use for classification in our neural network, a generic neural network model
    :param num_classes:
    :return:
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=(28,28)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model