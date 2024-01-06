from typing import NamedTuple
from pathlib import Path
from typing import Self
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import mnist

class PredictResult(NamedTuple):
    states: npt.NDArray[np.int8]
    energies: npt.NDArray[np.float32]
    
class HopfieldNet:
    """
    This is a fully connected HopfieldNet with the following properties:
    - inputs are assuemd to be 2-dimensional.
    - the number of units is the same as the dimension of an input.
    - the state is an 1-dimensional vector with size of the number of units.
    - a unit of the state is either -1 or 1.
    - the weight matrix is constructed by Hebbian learning.
    - the weight matrix is symmetric with zero-diagonal elements.
    - the bias term is the same for all units.
    - the activation function is sign function.
    """

    def __init__(self, inputs: npt.NDArray[np.int8]) -> None:
        self.dim = len(inputs[0])
        self.patterns = len(inputs)
        self.W = np.zeros((self.dim, self.dim))
        mean = np.sum([np.sum(i) for i in inputs]) / (self.patterns * self.dim)
        for i in range(self.patterns):
            # something like standardization
            t = inputs[i] - mean
            self.W += np.outer(t,t)
        for j in range(self.dim):
            self.W[j,j] = 0
        self.W /= self.patterns
    
    def energy(self, x: npt.NDArray[np.int8], bias: float) -> float:
        return -0.5 * np.dot(x.T, np.dot(self.W, x)) + np.sum(bias * x)

    def sync_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        es = [self.energy(x, bias)]
        xs = [x]
        for i in range(100):
            x_prev = xs[-1]
            e_prev = es[-1]
            x_new = np.sign(np.dot(self.W, x_prev) - bias)
            e_new = self.energy(x_new, bias)
            # if abs(e_new - e_prev) < 1e-7:
            #     return PredictResult(states=xs, energies=es)
            xs.append(x_new)
            es.append(e_new)
        return PredictResult(states=xs, energies=es)

    def async_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        es = [self.energy(x, bias)]
        xs = [x]
        for i in range(len(x)):
            state = xs[-1].copy()
            state_i_new = np.sign(np.dot(self.W[i,:], state) - bias)
            state[i] = state_i_new
            xs.append(state)
            es.append(self.energy(state, bias))
        return PredictResult(states=xs, energies=es)
    
    def predict(self, x: npt.NDArray[np.int8], bias: float, sync: bool) -> PredictResult:
        return self.sync_predict(x, bias) if sync else self.async_predict(x, bias)
    

def save_hopfield__mnist_prediction(fetch: mnist.MnistForHopfield, bias: float, sync: bool) -> None:
    train = fetch.original
    test = fetch.noised
    model = HopfieldNet(inputs=train)
    states = []
    energies = []
    for data_idx in range(len(train)):
        r = model.predict(train[data_idx], bias=bias, sync=sync)
        states.append([train[data_idx], test[data_idx], r.states[-1]])
        energies.append(r.energies)
    # show mnist image prediction
    n_patterns = len(states)
    n_transitions = len(states[0])
    if (n_patterns == 1):
        fig, ax = plt.subplots(1, n_transitions, figsize=(10, 5))
        for i in range(n_transitions):
            ax[i].matshow(states[0][i].reshape((28, 28)), cmap='gray')  
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            title = 'train_0' if i == 0 else 'test_0' if i == 1 else 'predict_0'
            ax[i].set_title(title)    
    else:
        fig, ax = plt.subplots(n_patterns, n_transitions, figsize=(10, 5))
        for j in range(n_patterns):
            for k in range(n_transitions):
                ax[j, k].matshow(states[j][k].reshape((28, 28)), cmap='gray')
                ax[j, k].set_xticks([])
                ax[j, k].set_yticks([])
                pfx = 'train' if k == 0 else 'test' if k == 1 else 'predict'
                sfx = f'_{j}'
                ax[j, k].set_title(pfx + sfx)
    prefix = 'sync' if sync else 'async'
    save_path = Path('./png') / f'{prefix}_mnist_image_prediction'
    plt.savefig(save_path)
    plt.close()
    # show energy transition
    x_axis = np.arange(len(energies[0]))
    for energy_idx in range(len(energies)):
        plt.plot(x_axis, energies[energy_idx], label=f'pattern_{energy_idx}')
    plt.legend()
    save_path = Path('./png') / f'{prefix}_energy_transition'
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    fetch = mnist.fetch_minist_for_hopfield(size=3, error_rate=0.14)    
    bias = 45
    save_hopfield__mnist_prediction(fetch=fetch, bias=bias, sync=True)
    save_hopfield__mnist_prediction(fetch=fetch, bias=bias, sync=False)
