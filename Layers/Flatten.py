import numpy as np


class Flatten:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor_shape = self.input_tensor.shape
        self.input_tensor = np.ravel(self.input_tensor).reshape(self.input_tensor.shape[0], -1)
        return self.input_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor = self.error_tensor.reshape(self.input_tensor_shape)
        return self.error_tensor
