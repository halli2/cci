"""Functions to calculate training metrics"""

from typing import Any

from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryROC,
)


def update_metrics(metrics: dict, prediction, y):
    for metric in metrics.values():
        metric.update(prediction, y)


def reset_metrics(metrics: dict):
    for metric in metrics.values():
        metric.reset()


def write_metrics(metrics: dict, loss: float, writer: SummaryWriter, epoch: int, cat: str):
    for key, metric in metrics.items():
        if key in ["roc", "conf"]:
            pass
        else:
            writer.add_scalar(f"{key}/{cat}", metric.compute(), epoch)
    writer.add_scalar(f"loss/{cat}", loss, epoch)


def metric_to_device(metrics, device):
    for metric in metrics.values():
        metric.to(device)


def create_metrics() -> dict[str, Any]:
    return {
        "conf": BinaryConfusionMatrix(),
        "acc": BinaryAccuracy(),
        "roc": BinaryROC(),
        "auroc": BinaryAUROC(),
        "f1": BinaryF1Score(),
    }
