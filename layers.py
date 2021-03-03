from activation_functions import Linear_Fun, ReLU, Tanh, Sigmoid, Leaky_ReLU, Swish
import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.uniform(-1, 1, (output_dim, input_dim))
        self.bias = np.random.uniform(-1, 1, (output_dim, 1))
        self.log = {}
        self.gradients = {}
        self.function = None

    def forward(self, x):
        self.log["x_prev"] = x
        self.log["a_next"] = self.weights @ x + self.bias
        self.log["x_next"] = self.function(self.log["a_next"])
        return self.log["x_next"]

    def backward(self, error):
        self.delta = error * self.function.derivative(self.log["x_next"])
        self.gradients["W"] = (self.delta @ self.log["x_prev"].T)
        self.gradients["b"] = self.delta
        error = self.weights.T @ self.delta
        return error

    def _update_weights(self, lr):
        self.weights -= lr * self.gradients["W"]
        self.bias -= lr * self.gradients["b"]


class Linear_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Linear_Fun()


class ReLU_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = ReLU()


class Tanh_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Tanh()


class Sigmoid_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Sigmoid()


class Leaky_ReLU_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Leaky_ReLU()

class Swish_Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Swish()
