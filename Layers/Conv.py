import numpy as np
from scipy import signal
from scipy.signal import correlate2d, convolve, convolve2d, correlate
import copy


class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = int(num_kernels)

        self.weights = np.random.uniform(0, 1, size=(self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, size=self.num_kernels)
        self.bias_optimizer = None
        self.weights_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        fan_in = np.product(self.convolution_shape)
        fan_out = np.product((self.num_kernels, *self.convolution_shape[1:]))

        self.weights = self.weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = self.bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        batch_shape = self.input_tensor.shape[0]
        channels = self.input_tensor.shape[1]
        height = self.input_tensor.shape[2]
        width = 0

        one_dimensional = False
        if len(self.input_tensor.shape) == 3:
            one_dimensional = True

        if one_dimensional:
            width = 0
        else:
            width = self.input_tensor.shape[3]

        if one_dimensional:
            self.pad = (self.convolution_shape[1] - 1, 0)
        else:
            self.pad = (self.convolution_shape[1] - 1, self.convolution_shape[2] - 1)

        if one_dimensional:
            self.pad_list = [(0, 0), (0, 0),
                             (np.ceil(self.pad[0] / 2).astype(int), np.floor(self.pad[0] / 2).astype(int))]
        else:
            self.pad_list = [(0, 0), (0, 0),
                             (np.ceil(self.pad[0] / 2).astype(int), np.floor(self.pad[0] / 2).astype(int)),
                             (np.ceil(self.pad[1] / 2).astype(int), np.floor(self.pad[1] / 2).astype(int))]

        self.input_tensor_pad = np.pad(self.input_tensor, self.pad_list, mode="constant", constant_values=(0, 0))

        if one_dimensional:
            self.output_tensor = (batch_shape, self.num_kernels, height)
        else:
            self.output_tensor = (batch_shape, self.num_kernels, height, width)

        self.output_tensor = np.zeros(self.output_tensor)
        self.output_tensor_shape = self.output_tensor.shape

        # Looping starts

        for i in range(batch_shape):
            for j in range(self.num_kernels):
                for k in range(channels):

                    correlation = 0
                    if one_dimensional:
                        correlation = signal.correlate(self.input_tensor_pad[i, k, :], self.weights[j, k, :], 'valid')
                        self.output_tensor[i, j, :] += correlation

                    else:
                        correlation = correlate2d(self.input_tensor_pad[i, k, :, :], self.weights[j, k, :, :], 'valid')
                        self.output_tensor[i, j, :, :] += correlation

                self.output_tensor[i, j, :] += self.bias[j]

        if one_dimensional:
            self.output_tensor = self.output_tensor[:, :, ::int(self.stride_shape[0])]
        else:
            self.output_tensor = self.output_tensor[:, :, ::int(self.stride_shape[0]), ::int(self.stride_shape[1])]

        return self.output_tensor

    @property
    def weights(self):
        return self.__gradient_weights

    @weights.setter
    def weights(self, weights):
        self.__gradient_weights = weights

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)
        self.weights_optimizer = copy.deepcopy(optimizer)

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        self.error_tensor_reshape = self.error_tensor.reshape(self.output_tensor.shape)
        self.error_tensor_weights = np.zeros(self.output_tensor_shape)

        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.num_kernels)

        self.previous_error = np.zeros(self.input_tensor.shape)
        self.previous_error_shape = np.zeros(self.input_tensor.shape)

        channel_i = self.input_tensor.shape[1]
        channel_e = self.error_tensor_reshape.shape[1]
        batch_e = self.error_tensor_reshape.shape[0]

        # padding
        one_dimensional = False
        if len(self.input_tensor.shape) == 3:
            one_dimensional = True

        if one_dimensional:
            self.pad = (self.convolution_shape[1] - 1, 0)
        else:
            self.pad = (self.convolution_shape[1] - 1, self.convolution_shape[2] - 1)

        if one_dimensional:
            self.pad_list = [(0, 0), (0, 0),
                             (np.ceil(self.pad[0] / 2).astype(int), np.floor(self.pad[0] / 2).astype(int))]
        else:
            self.pad_list = [(0, 0), (0, 0),
                             (np.ceil(self.pad[0] / 2).astype(int), np.floor(self.pad[0] / 2).astype(int)),
                             (np.ceil(self.pad[1] / 2).astype(int), np.floor(self.pad[1] / 2).astype(int))]

        self.input_tensor_pad = np.pad(self.input_tensor, self.pad_list, mode="constant", constant_values=(0, 0))

        for i in range(batch_e):
            for j in range(channel_e):

                self.gradient_bias[j] += np.sum(error_tensor[i, j, :])

                if one_dimensional:
                    self.error_tensor_weights[:, :, ::int(self.stride_shape[0])] = self.error_tensor_reshape
                else:
                    self.error_tensor_weights[:, :, ::int(self.stride_shape[0]), ::int(self.stride_shape[1])] = self.error_tensor_reshape

                for k in range(channel_i):
                    if one_dimensional:
                        self.previous_error[i, k, :] += convolve(self.error_tensor_weights[i, j, :], self.weights[j, k, :], 'same')
                    else:
                        self.previous_error[i, k, :, :] += convolve2d(self.error_tensor_weights[i, j, :, :], self.weights[j, k, :, :], 'same')

            for l in range(self.num_kernels):
                for k in range(channel_i):

                    if one_dimensional:
                        self.gradient_weights[l, k, :] += correlate(self.input_tensor_pad[i, k, :], self.error_tensor_weights[i, l, :], 'valid')
                    else:
                        self.gradient_weights[l, k, :, :] += correlate2d(self.input_tensor_pad[i, k, :, :], self.error_tensor_weights[i, l, :, :], 'valid')

        if self.bias_optimizer is not None and self.weights_optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.previous_error



