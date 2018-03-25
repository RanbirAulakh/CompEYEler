from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    """
    Main entrypoint for tensorflow
    :arg argv (sys.argv)
    """
    print('hi mom', argv)

if __name__ == "__main__":
      tf.app.run()
