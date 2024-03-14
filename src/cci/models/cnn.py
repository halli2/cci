from torch import Tensor, nn


class CNNModel(nn.Module):
    """Returns logits"""

    def __init__(self, nb_features=1):
        super().__init__()
        channels = [6, 12]
        kernel_size = [32, 32]
        padding = "same"
        self.nb_features = nb_features
        self.sequential = nn.Sequential(
            nn.Conv1d(nb_features, channels[0], kernel_size[0], padding=padding),
            nn.BatchNorm1d(channels[0]),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(channels[0], channels[1], kernel_size[1], padding=padding),
            nn.BatchNorm1d(channels[1]),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.AvgPool1d(3),
            # Output layer
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.nb_features != 1:
            raise ValueError("nb_features different from 1 (multivariate) is not implemented yet.")
        x = x.view(-1, 1, x.size(1))
        logits = self.sequential(x)
        return logits.view(-1)
