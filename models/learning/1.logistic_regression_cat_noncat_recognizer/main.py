"""
Main file to create simple (cat / non cat) recognizer.

File does this:
 1. Loads dataset
 2. Inspects dataset shapes, and then allows you to view train set
 3. Trains model
 4. Plots you learning curve, and then allows you to view model decisions on test set

You should run it with -v (visualize train dataset before model training) and with -c (check model decisions after learning)

P.s. And ye, i know about jupiter notebooks, this file's purpose is to feel the need in jupiter notebooks. Hope you'll have fun with my code :)
"""

import getopt
import sys

from models.logistic_regression_cat_noncat_recognizer.library.basic_functions import load_dataset
from models.logistic_regression_cat_noncat_recognizer.library.visualization import plot_image_from_dataset, plot_learning_curve, plot_model_decision
from models.logistic_regression_cat_noncat_recognizer.library.model import model

# ------- Load dataset, run parameters ------- #

options, arguments = getopt.getopt(sys.argv, 'v')
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# ------- Inspecting dataset ------- #

# Figure out dimensions and shapes of the problem
train_set_size = train_set_x_orig.shape[0]
test_set_size = test_set_x_orig.shape[0]
pixels_in_image = train_set_x_orig.shape[2]
print(f"train_set_size: {train_set_size}, test_set_size: {test_set_size}, pixels_in_image: {pixels_in_image}")

# If we pass -v as args - we'll start inspecting dataset
run_visualization = '-v' in arguments
while run_visualization:
    image_index = int(input(f"Input image index (0 ... {train_set_size - 1}) to visualize image, or -1 to exit: "))

    if image_index < 0 or image_index > train_set_size - 1:
        break

    print("To get to next image you should close pyplot window")
    plot_image_from_dataset(train_set_x_orig, train_set_y, classes, image_index)

# ------- Prepare dataset and train model ------- #

# Reshape dataset for linear vector
train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_set_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

# Normalize values. Algorithm will run better on normalized values
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

coefficients = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# ------- Look at results ------- #

# Look al learning curve
plot_learning_curve(coefficients)

# If we pass -c as args - we'll start inspecting model decisions
run_decisions_checks = '-c' in arguments
while run_decisions_checks:
    image_index = int(input(f"Input image index (0 ... {test_set_size - 1}) to check image, or -1 to exit: "))

    if image_index < 0 or image_index > test_set_size - 1:
        break

    print("To get to next image you should close pyplot window")
    plot_model_decision(test_set_x, test_set_y, classes, coefficients, pixels_in_image, image_index)

print("Finished working with model")
