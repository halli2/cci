from torch import Tensor, nn
from optuna import Trial


class CNNModel(nn.Module):
    """Returns logits"""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, channels, length]
        x = x.view(-1, 1, x.size(1))
        logits = self.sequential(x)
        return logits.view(-1)


def default_model() -> CNNModel:
    channels = [6, 12]
    kernel_size = [32, 32]
    padding = "same"
    layers = [
        nn.Conv1d(1, channels[0], kernel_size[0], padding=padding),
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
    ]
    return CNNModel(layers)


def suggest_model(trial: Trial, max_kernel_size=14) -> CNNModel:
    """Constructs a simple CNN Model with increasing channels and decreasing kernel size"""
    n_cnn = trial.suggest_int("n_cnn", 1, 3)

    kernel_size = max_kernel_size # Max kernel size in first layer
    padding = "same"
    channels_in = 1
    layers: list[nn.Module] = []
    for i in range(n_cnn):
        channels_out = trial.suggest_int(f"channels_{i}", channels_in, 128)
        kernel_size = trial.suggest_int(f"kernel_{i}", 1, kernel_size)
        dropout = trial.suggest_float(f"dropout_{i}", 0.1, 0.5)

        layers += [
            nn.Conv1d(channels_in, channels_out, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels_out),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.AvgPool1d() # TODO: Use avg pool?
        ]

        channels_in = channels_out

    # Global average pooling [batch_sisze, channels, length] -> [batch_Size, channels, 1]
    layers += [nn.AdaptiveAvgPool1d(1)]
    layers += [
        nn.Flatten(),
        nn.Linear(channels_in, 1),  # 1 class
    ]
    return CNNModel(layers)
