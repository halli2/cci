from torch import nn

from .cnn import CNNModel
from .mlp import MLPModel


class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


_all_ = ["MLPModel", "CNNModel", "LambdaModule"]
