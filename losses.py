class Loss():
    def __init__(self):
        self.log = {}
    def __call__(self, x, y):
        pass
    def backward(self):
        pass

class LossMSE(Loss):
    def __call__(self, t, y_hat):
        self.log["t"] = t
        self.log["y_hat"] = y_hat
        return 0.5 * (t - y_hat)**2

    def backward(self):
        return (self.log["y_hat"] - self.log["t"])
