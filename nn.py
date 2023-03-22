import pandas as pd
import numpy as np
import numdifftools as nd
import time
import warnings
from typing import Callable

warnings.filterwarnings('ignore')
np.random.seed(0)

def sigmoid(x):
    y = 1 / (1 + np.e ** (-x))
    return y


def sigmoid_derivative(x):
    y = sigmoid(x) * (1 - sigmoid(x))
    return y


def relu(x):
    y = x if x >= 0 else 0
    return y


def buffer(x):
    print('This is a buffer function.')
    return 0


class NeuralNetwork:
    def __init__(self, shape: tuple, act_funcs: list[Callable]):
        self.shape = shape
        self.act_funcs = [buffer] + act_funcs

        self.n_weights = np.dot(self.shape[:-1], self.shape[1:])
        self.n_biases = np.sum(self.shape[1:])
        self.n_parameters = self.n_weights + self.n_biases
        self.n_layers = len(self.shape)

        self.parameters = np.random.uniform(size=self.n_parameters)

        self._A = np.empty(shape=self.n_layers, dtype=object)
        self._W = np.empty(shape=self.n_layers, dtype=object)
        self._B = np.empty(shape=self.n_layers, dtype=object)

        self.expected = np.empty(shape=self.shape[-1])

    def feed_forward(self, food: np.ndarray):
        # don't worry, he doesn't byte
        self._A[0] = food

    @property
    def A(self):
        # self._A[0] = np.zeros(shape=self.shape[0])
        for i in range(1, self.n_layers):
            self._A[i] = self.act_funcs[i](np.dot(self.W[i], self._A[i-1]) + self.B[i])
        return self._A

    @A.setter
    def A(self, val):
        self._A = val

    @property
    def W(self):
        w_linear = self.parameters[:self.n_weights]
        indexes_of_slices = np.cumsum(np.multiply(self.shape[1:], self.shape[:-1]))
        w_slices = np.split(w_linear, indexes_of_slices)[:-1]
        for i, weights_slice in enumerate(w_slices):
            shape_new = (self.shape[i+1], self.shape[i])
            self._W[i+1] = np.reshape(weights_slice, shape_new)
        return self._W

    @W.setter
    def W(self, val):
        self._W = val

    @property
    def B(self):
        b_linear = self.parameters[:-self.n_biases]
        indexes_of_slices = np.cumsum(self.shape[1:])
        b_slices = np.split(b_linear, indexes_of_slices)[:-1]
        self._B[1:] = b_slices
        return self._B

    @B.setter
    def B(self, val):
        self._B = val

    def E(self, expected: np.ndarray):
        E = np.sum((self.A[-1] - expected) ** 2)
        return E

    def gradient(self):
        return 0

    def propagate_backward(self, expected: np.ndarray):
        self.expected = expected
        self.parameters += -self.gradient()

    def train(self, df: pd.DataFrame):
        df = df.apply(lambda x, y: (self.feed_forward(x), self.propagate_backward(y)))
        pass


class Test:
    def __init__(self, n: int):
        self.list = list(range(n))
        self.name = 'Petr'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, string):
        self._name = string

    @property
    def opp(self):
        return self.list[::-1]

    def change_list(self):
        self.list = None


neural_network = NeuralNetwork(shape=(43, 25, 13), act_funcs=[sigmoid, sigmoid])
neural_network.feed_forward(food=np.random.uniform(size=43))
print(neural_network.A[-1])
print(neural_network.E(expected=np.zeros(shape=13)))
