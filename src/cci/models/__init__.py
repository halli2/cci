from typing import Any, Callable

from torch import Tensor, nn

from .mlp import MLPModel


class LambdaLayer(nn.Module):
    def __init__(self, fn: Callable[[Any], Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x) -> Tensor:
        return self.fn(x)


_all_ = ["MLPModel", "LambdaLayer"]
