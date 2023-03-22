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

class Function:
    def __init__(self, func: Callable, derivative: Callable):
        self.func = func

    def __call__(self, x):
        return self.func(x)
    def

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
        self._A[0] = np.ones(shape=self.shape[0], dtype=object)

        self._W = np.empty(shape=self.n_layers, dtype=object)
        self._B = np.empty(shape=self.n_layers, dtype=object)
        self._E = None

        self.training_pair = np.empty(shape=self.shape[0]), np.empty(shape=self.shape[-1])
        self._food, self._expected = self.training_pair

    @property
    def A(self):
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

    @property
    def E(self):
        self._E = np.sum((self.A[-1] - self.expected) ** 2)
        return self._E

    @E.setter
    def E(self, val):
        self._E = val

    @property
    def food(self):
        return self.training_pair[0]

    @food.setter
    def food(self, val):
        self._food = val

    @property
    def expected(self):
        return self.training_pair[1]

    @expected.setter
    def expected(self, val):
        self._expected = val

    def gradient(self):
        return 0

    def feed_forward(self):
        # don't worry, he doesn't byte
        self._A[0] = self.food

    def propagate_backward(self):
        self.parameters += -self.gradient()

    def perform_iteration(self, data_pair: tuple[np.array]):
        self.training_pair = data_pair
        self.feed_forward()
        self.propagate_backward()

    def train(self, df: pd.DataFrame, n_iter: int = None):
        if n_iter is not None:
            df = df[df.index < n_iter]
        df.apply(self.perform_iteration, axis=1)


neural_network = NeuralNetwork(shape=(43, 25, 13), act_funcs=[sigmoid, sigmoid])

muffin = np.random.uniform(size=43)
pizza = np.random.uniform(size=43)
kebab = np.random.uniform(size=43)
apple = np.random.uniform(size=43)

exp1, exp2, exp3 = np.random.uniform(size=13), np.random.uniform(size=13), np.random.uniform(size=13)


data = pd.DataFrame({
    'food': (muffin, pizza, kebab, apple),
    'expected': (exp1, exp2, exp3, exp3)
})

print(neural_network.E)
neural_network.train(df=data)
print(neural_network.E)
