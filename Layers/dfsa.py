
def backward(self, error_tensor):
    # check if 1D
    one_dim = False
    if len(self.convolution_shape) == 2:
        one_dim = True
    print("Error tensor shape: ", error_tensor.shape)
    # Make shapes equal for error tensor and output from forward pass
    self.error_tensor_reshaped = error_tensor.reshape(self.output_tensor_unstrided.shape)
    self.error_upsampling = np.zeros(self.output_tensor_unstrided_shape)  # for wrt to weights
    self.gradient_bias = np.zeros(self.num_kernels)
    self.gradient_weights = np.zeros(self.weights.shape)
    print("Error tensor reshape: ", self.error_tensor_reshaped.shape)
    print("Error tensor unsampling: ", self.error_upsampling.shape)

    # gradient wrt layer shape
    prev_error = np.zeros(self.input_tensor.shape)  # for wrt to layer
    # get padding size
    if one_dim:
        self.padding = (self.convolution_shape[1] - 1, 0)
    else:
        self.padding = (self.convolution_shape[1] - 1, self.convolution_shape[2] - 1)

    # Create a list using ceil and floor
    if one_dim:
        self.padding_list = [(0, 0), (0, 0),
                             (np.ceil(self.padding[0] / 2).astype(int),
                              np.floor(self.padding[0] / 2).astype(int))]
    else:
        self.padding_list = [(0, 0), (0, 0),
                             (np.ceil(self.padding[0] / 2).astype(int),
                              np.floor(self.padding[0] / 2).astype(int)),
                             (np.ceil(self.padding[1] / 2).astype(int),
                              np.floor(self.padding[1] / 2).astype(int))]
    padded_tensor = np.pad(self.input_tensor, self.padding_list, mode="constant", constant_values=0)

    # over every batch of outputs
    for h in range(self.error_tensor_reshaped.shape[0]):
        # over every channel of outputs
        for s in range(self.error_tensor_reshaped.shape[1]):
            # error tensor runs over the batch.
            # gradient with respect to the bias is simply sums over En
            #
            self.gradient_bias[s] += np.sum(error_tensor[h, s, :])
            # error tensor is updated for each batch and each channel over each stride
            if one_dim:
                self.error_upsampling[:, :, ::int(self.stride_shape[0])] = self.error_tensor_reshaped
            else:
                self.error_upsampling[:, :, ::int(self.stride_shape[0]),
                ::int(self.stride_shape[1])] = self.error_tensor_reshaped

            # over channels of input tensor
            for c in range(self.input_tensor.shape[1]):
                if one_dim:
                    prev_error[h, c, :] += convolve(self.error_upsampling[h, s, :],
                                                    self.weights[s, c, :],
                                                    'same')
                else:
                    prev_error[h, c, :, :] += convolve2d(self.error_upsampling[h, s, :, :],
                                                         self.weights[s, c, :, :],
                                                         'same')

        for s in range(self.num_kernels):
            # over channels
            for s_k in range(self.input_tensor.shape[1]):
                if one_dim:
                    self.gradient_weights[s, s_k, :] += correlate(padded_tensor[h, s_k, :],
                                                                  self.error_upsampling[h, s, :], 'valid')
                else:
                    self.gradient_weights[s, s_k, :, :] += correlate2d(padded_tensor[h, s_k, :, :],
                                                                       self.error_upsampling[h, s, :, :], 'valid')

    if self._bias_optimizer is not None and self._weights_optimizer is not None:
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

    return prev_error

