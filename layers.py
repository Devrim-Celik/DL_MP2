from activation_functions import Linear_Fun, ReLU, Tanh, Sigmoid, Leaky_ReLU, Swish
import numpy as np

class Layer:
    """
    Basic Layer class, that is used as building block for constructing
    NeuralNetwork classes.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialization function.

        Args:
            input_dim (int):    Number of input nodes
            output_dim (int):   Number of output nodes
        """
        # initialzing weight and bias with random uniform noise
        self.weights = np.random.uniform(-1, 1, (output_dim, input_dim))
        self.bias = np.random.uniform(-1, 1, (output_dim, 1))
        # declaring parameters
        self.log = {}
        self.gradients = {}
        self.name = None
        self.function = None

    def forward(self, x):
        """
        Forward step by the layer.

        Args:
            x (nd.array):   Activation of previous layer
        Returns:
            (nd.array):     Activaiton of next layer
        """
        self.log["x_prev"] = x
        # calculate summed input for next layer
        self.log["a_next"] = self.weights @ x + self.bias
        # apply activation function
        self.log["x_next"] = self.function(self.log["a_next"])
        return self.log["x_next"]

    def backward(self, error):
        """
        Backward step by the layer.

        Args:
            error (nd.array):   Error passed from the suceeding layer
                        (last layer defined by:     y_hat - t)
                        (other hidden layers:       W.T @ error_signal)
        Returns:
            (nd.array):         Error passed to the previous layer
        """
        # use the error
        self.delta = error * self.function.derivative(self.log["x_next"])
        # calculate and save gradient for later weight update
        self.gradients["W"] = (self.delta @ self.log["x_prev"].T)
        self.gradients["b"] = self.delta
        # calcualte the error for the previous layer
        error = self.weights.T @ self.delta
        return error

    def weights(self):
        """
        Returning the current weight matrix and bias vector.

        Args:
            self.weights (nd.array):        Weight matrix
            self.bias (nd.array):           Bias vector
        """
        return self.weights, self.bias

    def _update_weights(self, lr):
        """
        Using the stored gradient, this function update the internal parameters
        according to GD.

        Args:
            lr (float):     Learning rate
        """
        self.weights -= lr * self.gradients["W"]
        self.bias -= lr * self.gradients["b"]

    def __repr__(self):
        return "[" + self.name + f" with shape {self.weights.shape}]"


class Linear_Layer(Layer):
    """
    Linear Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Linear_Fun()
        self.name = "Linear"

class ReLU_Layer(Layer):
    """
    ReLU Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = ReLU()
        self.name = "ReLU"

class Tanh_Layer(Layer):
    """
    Tanh Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Tanh()
        self.name = "Tanh"

class Sigmoid_Layer(Layer):
    """
    Sigmoid Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Sigmoid()
        self.name = "Sigmoid"

class Leaky_ReLU_Layer(Layer):
    """
    Leaky ReLU Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Leaky_ReLU()
        self.name = "Leaky_ReLU"

class Swish_Layer(Layer):
    """
    Swish Layer Object
    """
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.function = Swish()
        self.name = "Swish"
