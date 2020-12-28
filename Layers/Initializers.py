import numpy as np
import math


class Constant:

    def __init__(self, constant_weights):
        self.constant_weights = constant_weights
        self.constant_weights = 0.1

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.constant_weights = 0.1
        return self.constant_weights


class UniformRandom:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.uniform_weights = np.random.uniform(0, 1, (self.fan_in, self.fan_out))
        return self.uniform_weights


class Xavier:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out

        sigma = float(math.sqrt(2 / (self.fan_in + self.fan_out)))
        xavier_weights = np.random.normal(0, sigma, self.weights_shape)

        return xavier_weights


class He:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out

        sigma = float(math.sqrt(2 / self.fan_in))
        he_weights = np.random.normal(0, sigma, self.weights_shape)
        return he_weights
