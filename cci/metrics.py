"""Functions to calculate training metrics"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
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
        self.subset = subset
        self.fold = fold
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
                val = value.detach().cpu().numpy().tolist()
            else:
                val = value.detach().cpu().item()
            self.saved_metrics[key].append(val)
        self.saved_metrics["bac"].append(
            ((computed_metrics["recall"] + computed_metrics["specificity"]) / 2).detach().cpu().item()
        )
        self.saved_metrics["epoch"].append(epoch)

    def reset(self) -> None:
        self.collection.reset()
        self.running_loss = torch.tensor(0.0)

    def write_metrics(self, dir: Path) -> None:
        """Save the metrics as a csv file"""
        # Convert cm to 4 variables for saving as CSV
        df = (
            pl.LazyFrame(
                self.saved_metrics,
            )
            .with_columns(
                pl.col("cm").map_elements(lambda x: x[0][1], return_dtype=pl.Int32).alias("TP"),
                pl.col("cm").map_elements(lambda x: x[0][1], return_dtype=pl.Int32).alias("FP"),
                pl.col("cm").map_elements(lambda x: x[1][0], return_dtype=pl.Int32).alias("FN"),
                pl.col("cm").map_elements(lambda x: x[1][1], return_dtype=pl.Int32).alias("TN"),
            )
            .drop("cm")
            .collect()
        )
        df.write_csv(dir / f"{self.fold}_{self.subset}.csv")
        pass

    def best_metrics(self) -> dict[str, Any]:
        best_metrics: dict[str, Any] = {}
        for key, val in self.saved_metrics.items():
            if key in ["cm", "epoch"]:
                continue
            elif key == "loss":
                best_metrics[key] = min(val)
            else:
                best_metrics[key] = max(val)
        return best_metrics


def confusion_matrix(cm) -> go.Figure:
    normalized_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=["Good (0)", "Bad (1)"],
        y=["Good (0)", "Bad (1)"],
        text_auto=True,
    )
    return fig
