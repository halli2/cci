from utils import project_dir
import torch
from torch import nn, optim, tensor, Tensor
from torch.nn import functional as F
import structlog
from pathlib import Path
import numpy as np
from metrics import Metrics
from torch.utils.data import DataLoader
from typing import Any, Dict

RESULTS_DIR = project_dir() / "results"
DATA_DIR = project_dir() / "data"
SAMPLE_LENGTH = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Callback:
    def __init__(self):
        pass

    def on_training_start(self, **kwargs) -> None | dict[Any, Any]:
        pass

    def on_epoch_end(self, **kwargs) -> None | dict[Any, Any]:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=200):
        super().__init__()
        self.patience = patience
        self.best_epoch = 0.0
        self.best_loss = np.inf

    def on_epoch_end(self, last_metrics, epoch, **kwargs):
        loss = last_metrics["validation"]["loss"]
        if loss < self.best_loss:
            self.best_epoch = epoch
            self.best_loss = loss
        elif epoch - self.best_epoch > self.patience:
            return {"stop": True}


class ModelCheckPoint(Callback):
    def __init__(self, fpath: Path):
        super().__init__()
        self.best_loss = 0.0
        self.best_loss = np.inf
        self.fpath = fpath

    def on_epoch_end(self, last_metrics, model, **kwargs):
        loss = last_metrics["validation"]["loss"]
        if loss < self.best_loss:
            self.best_loss = loss
            torch.jit.script(model).save(self.fpath)
            # torch.save(model.state_dict(), self.fpath)


class EpochInfoLogger(Callback):
    def __init__(self, frequency=10):
        super().__init__()
        self.frequency = frequency

    def on_training_start(self, epochs, **kwargs):
        logger = structlog.get_logger()
        logger.info("Epochs", epochs=epochs)

    def on_epoch_end(self, last_metrics, epoch, **kwargs):
        logger = structlog.get_logger()
        training_values = last_metrics["training"]
        validation_values = last_metrics["validation"]
        if epoch % self.frequency == 0:
            logger.debug(
                "Epoch Info",
                epoch=epoch,
                training_accuracy=training_values["acc"],
                training_loss=training_values["loss"],
                validation_accuracy=validation_values["acc"],
                validaion_loss=validation_values["loss"],
            )


def fit(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.BCEWithLogitsLoss,
    train_metrics: Metrics,
    val_metrics: Metrics,
    train_loader: DataLoader,
    val_loader: DataLoader,
    callbacks: list[Callback],
    epochs: int,
) -> tuple[Metrics, Metrics, Dict[Any, Any]]:
    status = {"stop": False}
    val_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for callback in callbacks:
        cb_res = callback.on_training_start(epochs=epochs, model=model)
        if cb_res is not None:
            status.update(cb_res)

    for epoch in range(1, epochs + 1):
        train_metrics.reset()
        model.train()

        for data in train_loader:
            sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)
            batch_size = len(label)
            opt.zero_grad()
            logits = model(sample)

            loss = loss_fn(logits, label.float())
            loss.backward()
            opt.step()

            predictions = F.sigmoid(logits)
            train_metrics.update(predictions, label, loss * batch_size)
        train_metrics.save_metrics(epoch)

        # Track weights
        # track_params_dists(model, run)
        # track_gradients_dists(model, run)

        val_metrics.reset()
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)

                logits = model(sample)

                loss = val_loss_fn(logits, label.float())

                predictions = F.sigmoid(logits)
                val_metrics.update(predictions, label, loss)
        val_metrics.save_metrics(epoch)

        last_metrics = {"validation": val_metrics.compute(), "training": train_metrics.compute()}

        for callback in callbacks:
            cb_res = callback.on_epoch_end(last_metrics=last_metrics, epoch=epoch, model=model)
            if cb_res is not None:
                status.update(cb_res)

        if status["stop"]:
            break

    return train_metrics, val_metrics, status


class AutoModel(nn.Module):
    """sa wakanai"""

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1500, 1000, 5, batch_first=True)
        self.lstm2 = nn.LSTM(1000, 500, 5, batch_first=True)
        self.lstm3 = nn.LSTM(500, 5, 1, batch_first=True)
        self.fc = nn.Linear(5, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 1, x.size(1))
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x


def train_model(train_loader, val_loader):
    model: nn.Module = todo()
    opt = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    train_metrics = Metrics("train", len(train_loader.dataset), DEVICE, fold_idx)
    val_metrics = Metrics("val", len(val_loader.dataset), DEVICE, fold_idx)

    train_metrics, val_metrics, status = fit(
        model,
        opt,
        loss_fn,
        train_metrics,
        val_metrics,
        train_loader,
        val_loader,
        [
            EarlyStopping(patience=200),
            EpochInfoLogger(50),
            ModelCheckPoint(RESULTS_DIR / "tmp_model.pt"),
        ],
        500,
    )
