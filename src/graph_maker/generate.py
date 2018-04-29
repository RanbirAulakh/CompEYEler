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
from neural_network.keras import train_network

def get_accuracy(model):
    """
    Tests a model on a few defined training sets from the greedy segmenter
    :returns overallaccuracy plot, character_accuracy plot
    """
    pass

def find_best_epoch(min_epoch=1, max_epoch=1000):
    """
    Iterates over epochs min-max, produces graphs for all in terms of accuracy
    against the training set we have
    """
    for i in range(min_epoch, max_epoch):
        train_network(min_epoch)



def main():
    """
    Main entry point for application
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) != 2:
        print('usage: python train.py')
        sys.exit()
    find_best_epoch()

if __name__ == '__main__':
    main()
