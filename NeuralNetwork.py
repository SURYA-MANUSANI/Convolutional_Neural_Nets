import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []

    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()
        for i in range(len(self.layers)):
            self.input_tensor = self.layers[i].forward(self.input_tensor)
        forward_loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return forward_loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for j in reversed(range(len(self.layers))):
            self.error_tensor = self.layers[j].backward(self.error_tensor)

    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for k in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self,input_tensor):
        for l in range(len(self.layers)):
            input_tensor = self.layers[l].forward(input_tensor)
        return input_tensor

