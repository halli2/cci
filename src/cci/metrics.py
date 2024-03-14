"""Functions to calculate training metrics"""

from typing import Any, Dict

import aim
import numpy as np
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
    BinaryROC,
)


class Metrics:
    def __init__(self, subset: str, dataset_length: int, device, fold=0):
        self.collection = MetricCollection(
            {
                "cm": BinaryConfusionMatrix(normalize="true"),
                "acc": BinaryAccuracy(),
                "roc": BinaryROC(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "prcurve": BinaryPrecisionRecallCurve(),
                "auprc": BinaryAveragePrecision(),
            }
        ).to(device)
        self.context = {"subset": subset, "fold": fold}
        self.dataset_length = dataset_length
        self.running_loss = torch.tensor(0.0)
        self.best_metrics = {
            "acc": -np.inf,
            "loss": np.inf,
            "f1": -np.inf,
            "auroc": -np.inf,
            "precision": -np.inf,
            "recall": -np.inf,
            "auprc": -np.inf,
        }

    def to(self, device):
        self.collection.to(device)

    def update(self, predictions: Tensor, label: Tensor, loss: Tensor):
        self.collection.update(predictions, label)
        self.running_loss += loss.sum().to("cpu")

    def compute(self) -> Dict[str, Tensor]:
        vals: Dict[str, Tensor] = self.collection.compute()
        vals["loss"] = self.running_loss / self.dataset_length

        return vals

    def reset(self) -> None:
        self.collection.reset()
        self.running_loss = torch.tensor(0.0)

    def check_for_best(self, vals):
        self.best_metrics["acc"] = max(self.best_metrics["acc"], vals["acc"].item())
        self.best_metrics["loss"] = min(self.best_metrics["loss"], vals["loss"].item())
        self.best_metrics["f1"] = max(self.best_metrics["f1"], vals["f1"].item())
        self.best_metrics["auroc"] = max(self.best_metrics["auroc"], vals["auroc"].item())
        self.best_metrics["precision"] = max(self.best_metrics["precision"], vals["precision"].item())
        self.best_metrics["recall"] = max(self.best_metrics["recall"], vals["recall"].item())
        self.best_metrics["auprc"] = max(self.best_metrics["auprc"], vals["auprc"].item())

    def upload_metrics_epoch(
        self,
        run: aim.Run,
        epoch: int,
        plot_cm: bool = False,
    ) -> None:
        vals = self.compute()
        self.check_for_best(vals)
        for key, value in vals.items():
            if key in ["cm", "roc", "prcurve"]:
                continue

            run.track(value.to("cpu"), name=key, epoch=epoch, context=self.context)  # type: ignore
        if plot_cm:
            fig = aim.Figure(confusion_matrix(vals))
            run.track(fig, name="cm", epoch=epoch, context=self.context)  # type: ignore

    def upload_training_end(self, run: aim.Run, prefix: str = "best_"):
        for key, value in self.best_metrics.items():
            run.track(value, name=f"{prefix}{key}", context=self.context)  # type: ignore

    def upload_test(self, run: aim.Run):
        vals = self.compute()
        self.check_for_best(vals)
        self.upload_training_end(run, prefix="")
        fig = aim.Figure(confusion_matrix(vals))
        run.track(fig, name="cm", context=self.context)  # type: ignore


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
