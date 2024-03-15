"""Functions to calculate training metrics"""

from typing import Any, Dict

import torch
from plotly import express as px
from plotly import graph_objs as go
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,  # AreaUnderPrecisionRecallCurve
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinarySpecificity,
)


class Metrics:
    def __init__(self, subset: str, dataset_length: int, device, fold=0):
        self.collection = MetricCollection(
            {
                "cm": BinaryConfusionMatrix(),
                "acc": BinaryAccuracy(),
                # "roc": BinaryROC(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "specificity": BinarySpecificity(),
            }
        ).to(device)
        self.context = {"subset": subset, "fold": fold}
        self.dataset_length = dataset_length
        self.running_loss = torch.tensor(0.0)
        self.saved_metrics: dict[str, list[Any]] = {
            "epoch": [],
            "acc": [],
            "loss": [],
            "f1": [],
            "auroc": [],
            "precision": [],
            "recall": [],
            "cm": [],
            "specificity": [],
            "bac": [],
        }

    def to(self, device) -> None:
        self.collection.to(device)

    def update(self, predictions: Tensor, label: Tensor, loss: Tensor) -> None:
        self.collection.update(predictions, label)
        self.running_loss += loss.sum().to("cpu")

    def compute(self) -> Dict[str, Tensor]:
        vals: Dict[str, Tensor] = self.collection.compute()
        vals["loss"] = self.running_loss / self.dataset_length

        return vals

    def save_metrics(self, epoch: int) -> None:
        computed_metrics = self.compute()
        for key, value in computed_metrics.items():
            if key == "cm":
                val = value.detach().numpy().tolist()
            else:
                val = value.detach().item()
            self.saved_metrics[key].append(val)
        self.saved_metrics["bac"].append(
            ((computed_metrics["recall"] + computed_metrics["specificity"]) / 2).detach().item()
        )
        self.saved_metrics["epoch"].append(epoch)

    def reset(self) -> None:
        self.collection.reset()
        self.running_loss = torch.tensor(0.0)


def confusion_matrix(calculated_metrics: dict[str, Any], key: str = "cm") -> go.Figure:
    cm = calculated_metrics[key].to("cpu")
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=["Good (0)", "Bad (1)"],
        y=["Good (0)", "Bad (1)"],
        text_auto=True,
    )
    return fig
