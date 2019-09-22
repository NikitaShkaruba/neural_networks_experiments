"""
This file contains some helper visualization logic
"""

import getopt
import sys

import numpy as np
import matplotlib.pyplot as plt


def inspect_dataset_if_needed(train_set_x_orig, train_set_y, classes, train_set_size):
    options, arguments = getopt.getopt(sys.argv, 'v')
    run_visualization = '-v' in arguments
    while run_visualization:
        image_index = int(input(f"Input image index (0 ... {train_set_size - 1}) to visualize image, or -1 to exit: "))

        if image_index < 0 or image_index > train_set_size - 1:
            break

        print("To get to next image you should close pyplot window")
        plot_image_from_dataset(train_set_x_orig, train_set_y, classes, image_index)


def plot_image_from_dataset(set_x, set_y, classes, image_index):
    print("y = " + str(set_y[:, image_index]) + ", it's a '" + classes[np.squeeze(set_y[:, image_index])].decode("utf-8") + "' picture.")

    plt.imshow(set_x[image_index])
    plt.show(block=True)


def plot_model_decision(test_set_x, test_set_y, classes, coefficients, num_px, index):
    print(f"y = {str(test_set_y[0, index])}, model predicted that it is a {coefficients['y_prediction_test_matrix'][0, index]}% cat")

    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.show(block=True)


def plot_learning_curve(coefficients):
    costs = np.squeeze(coefficients['costs'])

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(coefficients["learning_rate"]))
    plt.show(block=True)
