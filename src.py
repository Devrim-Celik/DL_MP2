import pytorch as torch

class FC_Layer():
    def __init__(self, nr_neurons):
        self.layer = torch.empty(nr_neurons)

    def forward(self, x):
        return 0
    def backward(self):
        return 0
    def param(self):
        return
