import pandas as pd
import numpy as np
import numdifftools as nd
import time
import warnings

warnings.filterwarnings('ignore')


class NeuralNetwork:
    def __init__(self, shape: tuple, act_funcs: list[str]):
        self.shape = shape
        self.act_funcs = act_funcs

        self.n_weights = np.dot(self.shape[:-1], self.shape[1:])
        self.n_biases = np.sum(self.shape[1:])
        self.n_parameters = self.n_weights + self.n_biases

        self.A = None
        self.initialize_layers()

        self.parameters = None
        self.initialize_parameters()

        self.W = np.zeros(shape=len(self.shape)+1)

    def initialize_layers(self):
        a = np.empty(shape=len(self.shape), dtype=object)
        for i, layer_size in enumerate(self.shape):
            a[i] = np.zeros(shape=layer_size)
        self.A = a

    def initialize_parameters(self):
        self.parameters = np.random.uniform(size=self.n_parameters)

    @property
    def W(self):
        w_linear = self.parameters[self.n_weights:]
        indexes_of_slices = np.cumsum(np.multiply(self.shape[1:], self.shape[:-1]))
        w_slices = np.split(w_linear, indexes_of_slices)[:-1]
        # w_slices doesn't work
        _W = np.zeros(shape=len(self.shape)+1)
        for i, weights_slice in enumerate(w_slices):
            shape_new = (self.shape[i], self.shape[i+1])
            self._W[i+1] = np.reshape(weights_slice, shape_new)
        return self._W

    @W.setter
    def W(self, val):
        self._W = val

    @property
    def B(self):
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

bruh = Test(5)
print(bruh.list)
bruh.change_list()
print(bruh.list)
print(bruh.name)

neural_network = NeuralNetwork(shape=(43, 25, 14), act_funcs=['bruh'])
print(neural_network.W)