class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backpropagation(self, error, lr):
        for layer in reversed(self.layers):
            error = layer.backward(error)
        for layer in self.layers:
            layer._update_weights(lr)
