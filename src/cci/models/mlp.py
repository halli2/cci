from torch import Tensor, nn


class MLPModel(nn.Module):
    """Returns logits"""

    def __init__(self, nb_features=1500):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),  # Test with multivariate?
            nn.Dropout(0.1),
            nn.Linear(nb_features, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.sequential(x)
        return logits.view(-1)
