import pandas as pd
import numpy as np
import numdifftools as nd
import time
import warnings
from typing import Callable
from datetime import datetime, timedelta
import fbchat
from getpass import getpass

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
    # y = x if x >= 0 else 0
    return np.multiply(x, x >= 0)


def relu_derivative(x: np.array):
    # y = 1 if x >= 0 else 0
    return x >= 0


def softmax(x: np.array):
    x = x.astype(float)
    y = np.exp(x) / np.sum(np.exp(x))
    return y


def softmax_derivative(x: np.array):
    y = softmax(x) * (np.kron(x, x) - softmax(x))
    return y


def mean_square(predicted: np.array, expected: np.array):
    assert predicted.shape == expected.shape
    return (predicted - expected) ** 2


def mean_square_derivative(predicted: np.array, expected: np.array):
    assert predicted.shape == expected.shape
    return 2 * (predicted - expected)


def cross_entropy(predicted: np.array, expected: np.array):
    assert predicted.shape == expected.shape
    return -np.sum(expected * np.log(predicted))


def cross_entropy_derivative(predicted: np.array, expected: np.array):
    assert predicted.shape == expected.shape
    return -(expected / predicted) + (1 - expected) / (1 - predicted)


def buffer(x):
    print('This is a buffer function.')
    return 0


class NeuralNetwork:
    def __init__(self, shape: tuple, act_funcs: list[Callable], cost_func: Callable, eta: float = 0.0005,
                 eta_limit: float = 0.0, eta_factor: float = 2.0, adaptive_eta: bool = False,
                 continue_min_eta: bool = False, random_seed: int = 0):
        self.shape = shape
        self.act_funcs = [buffer] + act_funcs
        self.cost_func = cost_func

        self.derivative_dictionary = {
            relu: relu_derivative,
            sigmoid: sigmoid_derivative,
            softmax: softmax_derivative,
            mean_square: mean_square_derivative,
            cross_entropy: cross_entropy_derivative
        }

        self.n_weights = np.dot(self.shape[:-1], self.shape[1:])
        self.n_biases = np.sum(self.shape[1:])
        self.n_parameters = self.n_weights + self.n_biases
        self.n_layers = len(self.shape)

        np.random.seed(random_seed)
        self.parameters = np.random.uniform(size=self.n_parameters) - 0.5

        self._A = np.empty(shape=self.n_layers, dtype=object)
        self._A[0] = np.ones(shape=self.shape[0], dtype=object)

        self._W = np.empty(shape=self.n_layers, dtype=object)
        self._B = np.empty(shape=self.n_layers, dtype=object)

        self.eta = eta

        self.adaptive_eta = adaptive_eta
        self.eta_limit = eta_limit
        self.eta_factor = eta_factor
        self.eta_limit_warning_raised = False
        self.continue_min_eta = continue_min_eta

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
        b_linear = self.parameters[-self.n_biases:]
        indexes_of_slices = np.cumsum(self.shape[1:])
        b_slices = np.split(b_linear, indexes_of_slices)[:-1]
        self._B[1:] = b_slices
        return self._B

    @B.setter
    def B(self, val):
        self._B = val

    def E(self, expected: np.ndarray, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        # E = np.mean((self.A[-1] - expected) ** 2)
        E = self.cost_func(predicted=self.A[-1], expected=expected)
        return E

    def gradient(self, expected: np.array) -> np.array:
        def dE_dA(l: int):
            if l == self.n_layers - 1:
                # return 2 * (self.A[l] - expected)
                cost_func_derivative = self.derivative_dictionary[self.cost_func]
                return cost_func_derivative(predicted=self.A[l], expected=expected)
            else:
                return dE_dA(l+1) * dA_dZ(l+1) * self.W[l+1]

        def dE_dW(l: int):
            return np.outer(delta(l), self.A[l-1].T)

        def dE_dB(l: int):
            return delta(l)

        def dA_dZ(l: int):
            act_func_derivative = self.derivative_dictionary[self.act_funcs[l]]
            return act_func_derivative(self.A[l])
            # return act_func_derivative(np.dot(self.W[l], self.A[l-1]) + self.B[l])

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
        self.parameters += -self.eta * grad

    def perform_iteration(self, food: np.array, expected: np.array):
        self.feed_forward(food)
        return self.E(expected)

    def perform_epoch(self, df: pd.DataFrame, batch_size: int):
        df = df.sample(frac=1)
        n_batches = np.ceil(df.shape[0]/batch_size)
        for batch in np.array_split(df, n_batches):
            df = pd.DataFrame(batch)
            E_before = np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]), axis=1))
            parameters_before = self.parameters

            df.apply(lambda x: self.propagate_backward(expected=x.iloc[1]), axis=1)

            E_after = np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]), axis=1))
            if self.adaptive_eta:
                while E_after >= E_before:
                    if self.eta <= self.eta_limit and not self.eta_limit_warning_raised:
                        print('Reached learning rate limit!')
                        if self.continue_min_eta:
                            print('Continuing with eta_limit.')
                            self.eta = self.eta_limit
                            self.adaptive_eta = False
                            self.parameters = parameters_before
                            break
                        else:
                            print('Stopping learning cycle.')
                            self.eta_limit_warning_raised = True
                            self.parameters = parameters_before
                            break

                    self.parameters = parameters_before
                    self.eta = self.eta / self.eta_factor
                    print(f'Decreasing learning rate: eta = {self.eta:1.3e}')
                    df.apply(lambda x: self.propagate_backward(expected=x.iloc[1]), axis=1)
                    # test
                    E_after = np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]),
                                               axis=1))
                    print(E_after)
        E_after = np.mean(np.mean(df.apply(lambda x: self.perform_iteration(food=x.iloc[0], expected=x.iloc[1]),
                                               axis=1)))
        return E_after

    def train(self, df: pd.DataFrame, n_epochs: int, msg_freq: int = 15, batch_size: int = 32,
              alter_func: Callable = None):
        print('Starting training.')

        COLUMNS = ['start', 'stop', 'E_mean', 'eta', 'acc_train', 'acc_test', 'est_finish']
        df_stats = pd.DataFrame(columns=COLUMNS,
                                index=pd.RangeIndex(stop=n_epochs))

        start_train = datetime.now()
        last_msg = datetime.now()
        est_finish = None

        df0 = df
        for i in range(n_epochs):
            if alter_func is not None:
                df = alter_func(df0)

            if self.eta_limit_warning_raised:
                print(f'Convergence achieved, breaking training loop at epoch {i+1}/{n_epochs}.')
                break

            silence = datetime.now() - last_msg
            if silence >= timedelta(seconds=msg_freq):
                print()
                print(f'[{datetime.now()}]: Currently at epoch {i+1}/{n_epochs}')
                sec_per_epoch = ((datetime.now() - start_train) / (i+1)).total_seconds()
                print(f'Average duration per epoch: {sec_per_epoch:.2f} s.')

                est_finish = datetime.now() + timedelta(seconds=(n_epochs - i - 1)*sec_per_epoch)
                print(f'Estimated time of finishing: {est_finish}')
                print()
                last_msg = datetime.now()

            start = datetime.now()
            E_mean = self.perform_epoch(df, batch_size=batch_size)
            stop = datetime.now()
            acc_train = self.accuracy(df)

            df_stats.iloc[i] = {'start': start,
                                'stop': stop,
                                'E_mean': E_mean,
                                'acc_train': acc_train,
                                'acc_test': None,
                                'eta': self.eta,
                                'est_finish': est_finish}

        print('Training finished.')
        return df_stats

    def predict(self, food: np.array) -> np.array:
        self.feed_forward(food)
        return self.A[-1]

    def accuracy(self, df: pd.DataFrame):
        is_correct = df.apply(lambda x: np.argmax(self.predict(x.iloc[0])) == np.argmax(x.iloc[1]), axis=1)
        return np.mean(is_correct)

    def gradient_checking(self, expected: np.ndarray, theta: float = 1.e-4) -> float:
        e_2 = self.E(expected, parameters=self.parameters+theta)
        e_1 = self.E(expected, parameters=self.parameters-theta)
        return (e_2 - e_1) / (2 * theta)

