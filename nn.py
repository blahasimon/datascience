import pandas as pd
import numpy as np
import numdifftools as nd
import time
import warnings
from typing import Callable
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(0)


def sigmoid(x):
    x = x.astype(float)
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_derivative(x):
    y = sigmoid(x) * (1 - sigmoid(x))
    return y


def relu(x: np.array):
    y = x if x >= 0 else 0
    return y


def relu_derivative(x: np.array):
    y = 1 if x >= 0 else 0
    return y


def softmax(x: np.array):
    x = x.astype(float)
    y = np.exp(x) / np.sum(np.exp(x))
    return y


def softmax_derivative(x: np.array):
    y = softmax(x) * (np.kron(x) - softmax(x))
    return y


def buffer(x):
    print('This is a buffer function.')
    return 0


class Function:
    def __init__(self, func: Callable, derivative: Callable):
        self.func = func
        self.drv = derivative

    def __call__(self, x):
        return self.func(x)


class NeuralNetwork:
    def __init__(self, shape: tuple, act_funcs: list[Callable], eta: float = 0.0005):
        self.shape = shape
        self.act_funcs = [buffer] + act_funcs
        self.derivative_dictionary = {
            relu: relu_derivative,
            sigmoid: sigmoid_derivative,
            softmax: softmax_derivative
        }

        self.n_weights = np.dot(self.shape[:-1], self.shape[1:])
        self.n_biases = np.sum(self.shape[1:])
        self.n_parameters = self.n_weights + self.n_biases
        self.n_layers = len(self.shape)

        self.parameters = np.random.uniform(size=self.n_parameters)

        self._A = np.empty(shape=self.n_layers, dtype=object)
        self._A[0] = np.ones(shape=self.shape[0], dtype=object)

        self._W = np.empty(shape=self.n_layers, dtype=object)
        self._B = np.empty(shape=self.n_layers, dtype=object)

        self.eta = eta
        self.ETA_LIMIT = 1.e-6
        self.ETA_FACTOR = 1.5
        self.eta_limit_warning_raised = False

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

    def E(self, expected: np.ndarray):
        E = np.mean((self.A[-1] - expected) ** 2)
        return E

    def gradient(self, expected: np.array) -> np.array:
        def dE_dA(l: int):
            if l == self.n_layers - 1:
                return 2 * (self.A[l] - expected)
            else:
                return dE_dA(l+1) * dA_dZ(l+1) * self.W[l+1]

        def dE_dW(l: int):
            return np.outer(delta(l), self.A[l-1].T)

        def dE_dB(l: int):
            return delta(l)

        def dA_dZ(l: int):
            act_func_derivative = self.derivative_dictionary[self.act_funcs[l]]
            return act_func_derivative(self.A[l])

        def delta(l: int):
            if l == self.n_layers - 1:
                delta_l = np.multiply(dE_dA(l), dA_dZ(l))
            else:
                delta_l = np.dot(delta(l+1).T, self.W[l+1])
                delta_l = np.multiply(delta_l, dA_dZ(l))
            return delta_l

        grad_weights = np.concatenate([dE_dW(l).flatten() for l in range(1, self.n_layers)])
        grad_biases = np.concatenate([dE_dB(l).flatten() for l in range(1, self.n_layers)])
        grad = np.concatenate([grad_weights, grad_biases])
        return grad

    def feed_forward(self, food: np.array):
        # don't worry, he doesn't byte
        self._A[0] = food

    def propagate_backward(self, expected: np.array):
        grad = self.gradient(expected)
        grad /= np.linalg.norm(grad)

        # save current parameters
        _parameters = self.parameters

        # add gradient to create new parameters
        self.parameters += -self.eta * np.linalg.norm(self.parameters) * grad

    def perform_iteration(self, food: np.array, expected: np.array):
        self.feed_forward(food)
        return self.E(expected)

    def perform_epoch(self, df: pd.DataFrame):
        E_before = np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]), axis=1))
        parameters_before = self.parameters

        df.apply(lambda x: self.propagate_backward(expected=x.iloc[1]), axis=1)

        E_after = np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]), axis=1))

        while E_after >= E_before:
            if self.eta <= self.ETA_LIMIT and not self.eta_limit_warning_raised:
                print('Reached learning rate limit!')
                self.eta_limit_warning_raised = True
                self.parameters = parameters_before
                break

            self.parameters = parameters_before
            self.eta = self.eta / self.ETA_FACTOR
            df.apply(lambda x: self.propagate_backward(expected=x.iloc[1]), axis=1)

        return E_after

    def train(self, df: pd.DataFrame, n_epochs: int):
        costs = []
        for i in range(n_epochs):
            if self.eta_limit_warning_raised:
                print(f'Convergence achieved, breaking training loop at epoch {i+1}/{n_epochs}.')
                break

            st = time.time()
            E = np.mean(self.perform_epoch(df))
            costs.append(np.mean(E))
            print(f'Epoch {i+1}/{n_epochs} finished in {time.time() - st:.2f} seconds', end='\r')
        return costs

    def predict(self, food: np.array):
        self.feed_forward(food)
        return self.A[0]


muffin = np.random.uniform(size=43)
pizza = np.random.uniform(size=43)
kebab = np.random.uniform(size=43)
apple = np.random.uniform(size=43)

exp1, exp2, exp3 = np.random.uniform(size=13), np.random.uniform(size=13), np.random.uniform(size=13)

data = pd.DataFrame({
    'food': (muffin, pizza, kebab, apple),
    'expected': (exp1, exp2, exp3, exp3)
})

neural_network = NeuralNetwork(shape=(43, 25, 13), act_funcs=[sigmoid, sigmoid])

_ = []
N = 3000

_ = neural_network.train(df=data, n_epochs=N)
print()

'''
for i in range(N):
    _.append(np.mean(neural_network.perform_epoch(df=data)))
'''

plt.plot(range(len(_)), _)
plt.show()
