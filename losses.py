import numpy as np

class Loss():
    """
    Base Loss function to inherit from.
    """
    def __init__(self):
        self.log = {}
        self.name = None
    def __call__(self, x, y):
        pass
    def backward(self):
        pass
    def __repr__(self):
        return self.name

class LossMSE(Loss):
    """
    Mean squared error loss function.
    """
    def __call__(self, t, y_hat):
        self.log["t"] = t
        self.log["y_hat"] = y_hat
        return np.mean(0.5 * (t - y_hat)**2)

    def backward(self):
        """
        Derivative in regard to weight vector.

        Returns:
            (nd.array):     y_hat - t
        """
        return (self.log["y_hat"] - self.log["t"])
