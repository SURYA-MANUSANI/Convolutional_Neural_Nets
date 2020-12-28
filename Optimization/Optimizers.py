import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = float(learning_rate)
        self.v = 0
        self.momentum_rate = momentum_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.v * self.momentum_rate - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = float(learning_rate)
        self.v = 0
        self.r = 0
        self.mu = mu
        self.rho = rho
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.square(gradient_tensor)
        v_correction = self.v / (1 - np.power(self.mu, self.k))
        r_correction = self.r / (1 - np.power(self.rho, self.k))
        weight_tensor = weight_tensor - self.learning_rate * (
                    v_correction / (np.sqrt(r_correction) + np.finfo(float).eps))
        self.k = self.k + 1
        return weight_tensor