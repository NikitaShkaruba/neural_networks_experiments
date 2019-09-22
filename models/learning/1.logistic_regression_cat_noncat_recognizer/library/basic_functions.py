"""
This file contains some basic functions for neural networks building
"""

import math
import numpy as np
import h5py


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))

    return s


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    s_der = s * (1 - s)

    return s_der


def image2vector(image):
    v = image.reshape((1, image.shape[0] * image.shape[1] * image.shape[2]))

    return v


def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    normalized_x = x / x_norm

    return normalized_x


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum

    return s


def loss_1(y_hat, y):
    loss = np.sum(np.abs(y - y_hat))

    return loss


def loss_2(y_hat, y):
    diff = y - y_hat
    loss = np.sum(np.dot(diff, diff))

    return loss


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
