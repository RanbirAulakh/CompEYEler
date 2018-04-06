from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from classifier import get_data

# Imports
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    with tf.Session() as sess:
        # Get Image data and labels
        images, characters = get_data()

        # Initialize placeholders
        x = tf.placeholder(dtype=tf.float32, shape=[None, 20, 20])
        y = tf.placeholder(dtype=tf.int32, shape=[None])

        # Flatten the input data
        images_flat = tf.contrib.layers.flatten(x)

        # Fully connected layer
        logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

        # Define a loss function
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                             logits=logits))
        # Define an optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Convert logits to label indexes
        correct_pred = tf.argmax(logits, 1)

        # Define an accuracy metric
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        for i in range(201):
            _, loss_value = sess.run([train_op, loss], feed_dict={x: images, y: characters})
            if i % 10 == 0:
                print("Loss: ", loss)


if __name__ == "__main__":
      tf.app.run()
