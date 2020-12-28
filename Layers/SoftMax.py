import numpy as np


class SoftMax:
    def init(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        arr = np.exp(self.input_tensor - np.max(self.input_tensor))
        self.probabilites = arr / np.array([arr.sum(axis=1)]).T
        return self.probabilites

    def backward(self, error_tensor):
        error_tensor = self.probabilites * (
                error_tensor - np.sum(np.multiply(error_tensor, self.probabilites), axis=1, keepdims=True))
        return error_tensor
