import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (self.input_size, self.output_size))
        self.optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.bias = np.ones((1, self.output_size))
        self.output_tensor = np.dot(self.input_tensor, self.weights) + self.bias
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.previous_error = np.dot(self.error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, self.error_tensor)
        self.gradient_bias = np.sum(self.error_tensor, axis=0).reshape(1, self.error_tensor.shape[1])

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.previous_error

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.weights = self.weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.bias = self.bias_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
