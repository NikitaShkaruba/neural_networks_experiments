"""
This file contains model logic
"""

import numpy as np

from models.logistic_regression_cat_noncat_recognizer.library.basic_functions import sigmoid


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, x_matrix, y_matrix):
    """
    Implementation of the propagation

    Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        x_matrix -- data of size (num_px * num_px * 3, number of examples)
        y_matrix -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
        gradients - dictionary with dw, db
        cost -- negative log-likelihood cost for logistic regression
    """

    m = x_matrix.shape[1]

    # Forward propagation. From x to cost
    a_matrix = sigmoid(np.dot(w.T, x_matrix) + b)
    cost = -(1 / m) * np.sum(y_matrix * np.log(a_matrix) + (1 - y_matrix) * np.log(1 - a_matrix))

    # Backward propagation. To find gradient
    dw = (1 / m) * np.dot(x_matrix, (a_matrix - y_matrix).T)
    db = (1 / m) * np.sum(a_matrix - y_matrix)

    cost = np.squeeze(cost)

    gradients = {
        "dw": dw,
        "db": db
    }

    return gradients, cost


def optimize(w, b, x_matrix, y_matrix, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param x_matrix: data of shape (num_px * num_px * 3, number of examples)
    :param y_matrix: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps

    :returns:
        params: dictionary containing the weights w and bias b
        gradients: dictionary containing the gradients of the weights and bias with respect to the cost function
        costs: list of all the costs computed during the optimization, this will be used to plot the learning curve
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        gradients, cost = propagate(w, b, x_matrix, y_matrix)
        dw = gradients["dw"]
        db = gradients["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }
    gradients = {
        "dw": dw,
        "db": db
    }

    return params, gradients, costs


def predict(w, b, x_matrix):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param x_matrix: data of size (num_px * num_px * 3, number of examples)

    :returns: y_matrix_prediction - a numpy array (vector) containing all predictions (0/1) for the examples in x_matrix
    """

    y_matrix_prediction = np.zeros((1, x_matrix.shape[1]))
    w = w.reshape(x_matrix.shape[0], 1)

    # Compute vector "a_matrix" predicting the probabilities of a cat being present in the picture
    a_matrix = sigmoid(np.dot(w.T, x_matrix) + b)

    for i in range(a_matrix.shape[1]):
        # Convert probabilities a_matrix[0,i] to actual predictions p[0,i]
        y_matrix_prediction[0][i] = 1 if a_matrix[0, i] > 0.5 else 0

    return y_matrix_prediction


def model(x_train_matrix, y_train_matrix, x_test_matrix, y_test_matrix, num_iterations=2000, learning_rate=0.5,
          print_cost=False):
    """
    Builds the logistic regression model

    :param x_train_matrix: training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    :param y_train_matrix: training labels represented by a numpy array (vector) of shape (1, m_train)
    :param x_test_matrix: test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    :param y_test_matrix: test labels represented by a numpy array (vector) of shape (1, m_test)
    :param num_iterations: hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: hyperparameter representing the learning rate used in the update rule of optimize()
    :param print_cost: Set to true to print the cost every 100 iterations

    :returns coefficients: dictionary containing information about the model.
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(x_train_matrix.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, x_train_matrix, y_train_matrix, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    y_prediction_test_matrix = predict(w, b, x_test_matrix)
    y_prediction_train_matrix = predict(w, b, x_train_matrix)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train_matrix - y_train_matrix)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test_matrix - y_test_matrix)) * 100))

    coefficients = {
        "costs": costs,
        "y_prediction_test_matrix": y_prediction_test_matrix,
        "y_prediction_train_matrix": y_prediction_train_matrix,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return coefficients
