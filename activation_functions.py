import numpy as np

class Function:
    def __init__(self, *args, **kwargs):
        self.memory = {}
        self.gradient = {}

    def __call__(self, *args, **kwargs):
        pass

    def derivative(self, *args, **kwargs):
        pass


class Linear_Fun(Function):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class ReLU(Function):
    def __call__(self, x):
        x[x < 0] = 0
        return x

    def derivative(self, x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


class Tanh(Function):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        x = 1 - self(x)**2
        return x


class Sigmoid(Function):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x)*(1-self(x))


class Leaky_ReLU(Function):
    def __call__(self, x, c=0.01):
        self.c = c
        x[x < 0] *= c
        return x

    def derivative(self, x):
        x[x < 0] = self.c
        x[x > 0] = 1
        return x

class Swish(Function):
    def __call__(self, x, beta=1):
        self.beta = beta
        return x * self._sigmoid(beta * x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.beta * self(self.beta * x) + self._sigmoid(self.beta * x) * (1 - self.beta * self(self.beta * x))
