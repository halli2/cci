from typing import Any, Callable

from torch import Tensor, nn


class LambdaLayer(nn.Module):
    def __init__(self, fn: Callable[[Any], Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x) -> Tensor:
        return self.fn(x)


def reset_module_weights(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        if hasattr(module, "children"):
            for child in module.children():
                reset_module_weights(child)
