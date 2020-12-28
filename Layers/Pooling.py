import numpy as np


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_shape_f = self.input_tensor.shape[0]
        channels_f = self.input_tensor.shape[1]
        height_f = self.input_tensor.shape[2]
        width_f = self.input_tensor.shape[3]

        new_height = ((height_f - self.pooling_shape[0]) // self.stride_shape[0]) + 1
        new_width = ((width_f - self.pooling_shape[1]) // self.stride_shape[1]) + 1

        self.output_tensor = np.zeros((batch_shape_f, channels_f, new_height, new_width))

        for i in range(batch_shape_f):
            for j in range(channels_f):
                for k in range(new_height):
                    for l in range(new_width):

                        y_dir_start = k * self.stride_shape[0]
                        y_dir_end = k * self.stride_shape[0] + self.pooling_shape[0]
                        x_dir_start = l * self.stride_shape[1]
                        x_dir_end = l * self.stride_shape[1] + self.pooling_shape[1]

                        pool_layer = self.input_tensor[i, j, y_dir_start:y_dir_end, x_dir_start:x_dir_end]

                        self.pos_max_value = np.unravel_index(np.argmax(pool_layer), pool_layer.shape)
                        self.output_tensor[i, j, k, l] = np.max(pool_layer)

        return self.output_tensor

    def backward(self, error_tensor):

        self.error_tensor = error_tensor
        stored_pos = self.pos_max_value

        batch_shape_b = self.error_tensor.shape[0]
        channels_b = self.error_tensor.shape[1]
        height_b = self.error_tensor.shape[2]
        width_b = self.error_tensor.shape[3]

        self.previous_error = np.zeros(self.input_tensor.shape)

        for i in range(batch_shape_b):
            previous_input = self.input_tensor[i]
            for j in range(channels_b):
                for k in range(height_b):
                    for l in range(width_b):

                        y_dir_start = k * self.stride_shape[0]
                        y_dir_end = y_dir_start + self.pooling_shape[0]
                        x_dir_start = l * self.stride_shape[1]
                        x_dir_end = x_dir_start + self.pooling_shape[1]

                        pool_layer = previous_input[j, y_dir_start:y_dir_end, x_dir_start:x_dir_end]
                        mask = pool_layer == np.max(pool_layer)
                        self.previous_error[i, j, y_dir_start:y_dir_end, x_dir_start:x_dir_end] += np.multiply(mask, self.error_tensor[i, j, k, l])

        return self.previous_error