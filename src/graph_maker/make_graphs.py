"""
This file creates the actual graphs that we use for visually seeing the epoch
difference and accuracy as we change parameters

@author Joe Bartelmo

"""
import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def graph_all_data_epochs():
    """
    Takes in all dataepoch files and converts them to graphs to use
    :returns overallaccuracy plot, character_accuracy plot
    """
    total = 0
    total_no_space = 0
    with open("output.txt", "r") as f:
        EXPECTED = f.read()
        EXPECTED = EXPECTED[0:len(EXPECTED)-1]
        EXPECTED_MAP = {}
        for character in EXPECTED:
            if character not in EXPECTED_MAP:
                EXPECTED_MAP[character] = 1
            else:
                EXPECTED_MAP[character] = \
                        EXPECTED_MAP[character] + 1
            if character != " ":
                total_no_space += 1
            total += 1
    epochs = []
    accuracy = []
    accuracy_no_space = []
    character_accuracy = {}
    all_character_accuracy = []
    for i in os.listdir("./"):
        if os.path.isfile(os.path.join("./",i)) and 'dataepoch' in i:
            num_epochs = int(i[9:])
            epochs.append(num_epochs)
            my_char_acc = {}
            my_total_no_space = 0
            with open(i, 'r') as f:
                next(f)
                for line in f:
                    split=line.split('\t')
                    character = split[0]
                    count = int(split[1])
                    if character not in character_accuracy:
                        character_accuracy[character] = 0
                    if character != " ":
                        my_total_no_space += count
                    my_char_acc[character] = count
                    character_accuracy[character] = character_accuracy[character] + count
                f.seek(0)
                accuracy.append(float(f.readline()))
            accuracy_no_space.append(float(my_total_no_space)/float(total_no_space))
            all_character_accuracy.append(my_char_acc)
    for key in character_accuracy:
        character_accuracy[key] /= (EXPECTED_MAP[key]*float(len(epochs)))

    # so we don't graph something that looks like we had a seizure
    lists = sorted(zip(*[epochs, accuracy]))
    new_epoch, new_accuracy = list(zip(*lists))
    lists = sorted(zip(*[epochs, accuracy_no_space]))
    new_epoch, new_accuracy_no_spaces = list(zip(*lists))

    #now we just have to put this data into an intelligible form
    plt.plot(new_epoch, new_accuracy)
    plt.plot(new_epoch, new_accuracy_no_spaces)
    plt.legend(['Accuracy', 'Accuracy without Spaces'])
    plt.tight_layout()
    plt.show()
    plt.clf()

    #plot the character accuracy
    # we're going to do 1, 5, 10, 100
    lists = sorted(zip(*[epochs, all_character_accuracy]))
    new_epoch, new_accs = list(zip(*lists))
    for i in [1,5,10,100]:
        chars = []
        accuracy = []
        for key in new_accs[i]:
            chars.append(key)
            accuracy.append(\
                    float(new_accs[i][key]) / float(EXPECTED_MAP[key]))
        clists = sorted(zip(*[chars, accuracy]))
        new_chars, new_acc = list(zip(*clists))

        plt.bar(new_chars, new_acc)
        plt.tight_layout()
        plt.xlabel("Characters")
        plt.ylabel("Accuracy")
        plt.title("Epoch " + str(i))
        plt.show()



    

    return epochs, accuracy, accuracy_no_space, all_character_accuracy
                    




def main():
    """
    Main entry point for application
    """
    import sys
    if len(sys.argv) != 1:
        print('usage: python graph_maker.py')
        sys.exit()
    print(graph_all_data_epochs())

if __name__ == '__main__':
    main()
