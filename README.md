# HopfieldNet

Some applications of Hopfield Network are deposited in this repository.

## MNIST
It included two files (mnist.py and hopfield4gif.py).

mnist.py implements some functions to get and corrupt [the MNIST data](http://yann.lecun.com/exdb/mnist/) by making use of [scikit-learn](http://scikit-learn.org/stable/).

On the other hand, hopfield4gif.py implements both training and inferring algorithms (outer product construction and synchronous update rule). Given the trained data (MNIST handwritten digits) and the bias term, one can determine all parameters of a Hopfield network that reconstructs the trained data from the corrupted data.

The main function outputs a collection of the reconstructed data (80 png images) parametrized by the bias term. These png images are used to make a gif animation parametrized by the bias term.

<img src=MNIST/anim.gif width=400px>
