import numpy as np


class CrossEntropyLoss:
    def _init_(self):
        pass

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        log_likelihood = -np.log(self.input_tensor[np.where(label_tensor == 1)] + np.finfo(float).eps)
        loss = np.sum(log_likelihood)
        return loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        self.error_tensor = -self.label_tensor / self.input_tensor
        return self.error_tensor
