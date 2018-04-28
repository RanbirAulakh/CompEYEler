from classifier import get_data
from neural_networks import cnn_model_fn
import keras

def main():
    images, labels = get_data()
    model = cnn_model_fn(1000)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    #train
    model.fit(images, labels,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

if __name__ == "__main__":
      main()
