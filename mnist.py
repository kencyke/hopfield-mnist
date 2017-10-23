# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata

def mnist4hn(n_patterns):
    """

    Downloads the MNIST data from the mldata.org by using sklearn and
    randomly-extracts neural patterns of 784-dimensional vector with binary units
    from 70 thousand 28*28=784-dimensional vectors with 1 byte units.
    
    """
    mnist = fetch_mldata('MNIST original', data_home="~/Desktop/v3/")
    mnist.data = mnist.data.astype(np.float32)
    mnist.data /= 255
    #perm = np.random.permutation(mnist.data)
    #train = [perm[i] for i in range(n_patterns)]
    train = [mnist.data[11111], mnist.data[22222], mnist.data[33333]]
    train = [np.sign(t * 2 - 1) for t in train]
    return train

def addnoise(train, error_rate):
    """
    Adds random noise to the train data and returns it as the test data.
    Noise is added by flipping the sign of some units with the error rate p.

    """
    test = np.copy(train) # cf. from copy import copy/deepcopy
    for i, t in enumerate(test):
        s = np.random.binomial(1, error_rate, len(t))
        for j in range(len(t)):
            if s[j] != 0:
                t[j] *= -1
    return test

if __name__ == '__main__':
    # Set parameters
    n_patterns = 3
    n_units = 28
    error_rate = 0.1

    # Show the train data
    train = mnist4hn(n_patterns)
    fig, ax = plt.subplots(1, n_patterns, figsize=(10, 5))
    for i in range(n_patterns):
        ax[i].matshow(train[i].reshape((n_units, n_units)), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()

    # Show the test data
    test = addnoise(train, error_rate)
    fig, ax = plt.subplots(1, n_patterns, figsize=(10, 5))
    for i in range(n_patterns):
        ax[i].matshow(test[i].reshape((n_units, n_units)), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
