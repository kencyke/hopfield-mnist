# hopfield-mnist

It includes two python files (mnist.py and hopfield4gif.py).

mnist.py implements some functions to get and corrupt [the MNIST data](http://yann.lecun.com/exdb/mnist/) by making use of [scikit-learn](http://scikit-learn.org/stable/).

On the other hand, hopfield4gif.py implements both training and inferring algorithms (i.e., outer product construction and synchronous update rule). Given the trained data (i.e., the MNIST handwritten digits) and the bias term, one can determine all parameters of a Hopfield network that reconstructs the trained data from the corrupted data.

The main function outputs a collection of the reconstructed data (80 png images) parametrized by the bias term. These png images are used to make a gif animation parametrized by the bias term.

<img src=anim.gif width=400px>
