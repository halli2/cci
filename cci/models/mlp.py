import optuna
from torch import Tensor, nn

from . import LambdaLayer


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


def suggest_model(trial: optuna.Trial) -> list[nn.Module]:
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    i = 1
    layers: list[nn.Module] = [
        nn.Flatten(),
        nn.Dropout(trial.suggest_float(f"dropout_{i}", 0.1, 0.5)),
    ]
    features = trial.suggest_int(f"linear_{i}", 1, 1000)
    prev_features = features
    layers.append(nn.LazyLinear(features))
    layers.append(nn.ReLU())
    for _ in range(n_hidden_layers - 1):
        i += 1
        features = trial.suggest_int(f"linear_{i}", 1, 1000)

        layers.append(nn.Dropout(trial.suggest_float(f"dropout_{i}", 0.1, 0.5)))
        layers.append(nn.Linear(prev_features, features))
        layers.append(nn.ReLU())
        prev_features = features
    i += 1
    layers.append(nn.Dropout(trial.suggest_float(f"dropout_{i}", 0.1, 0.5)))
    layers.append(nn.Linear(prev_features, 1))
    layers.append(LambdaLayer(lambda x: x.view(-1)))
    return layers
