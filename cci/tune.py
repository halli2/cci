import json
import structlog
from typing import Any, Dict

import numpy as np
import optuna
from optuna.storages import JournalFileStorage, JournalStorage
import torch
from dataset import skfold
from deepdiff import DeepHash
from metrics import Metrics
from models import mlp, cnn
from dataset import TransitionDataset
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import polars as pl
from utils import project_dir
from dataset import CropSample
from pathlib import Path
from models.layers import reset_module_weights

RESULTS_DIR = project_dir() / "results"
DATA_DIR = project_dir() / "data"
SAMPLE_LENGTH = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Callback:
    def __init__(self):
        pass

    def on_training_start(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
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
            torch.save(model.state_dict(), self.fpath)


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
            logger.debug("Epoch", epoch=epoch)
            logger.debug("Training", accuracy=training_values["acc"], loss=training_values["loss"])
            logger.debug("Validation", accuracy=validation_values["acc"], loss=validation_values["loss"])


def fit(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.BCEWithLogitsLoss,
    val_loss_fn: nn.BCEWithLogitsLoss,
    train_metrics: Metrics,
    val_metrics: Metrics,
    train_loader: DataLoader,
    val_loader: DataLoader,
    callbacks: list[Callback],
    epochs: int,
) -> tuple[Metrics, Metrics, Dict[Any, Any]]:
    status = {"stop": False}
    for callback in callbacks:
        cb_res = callback.on_training_start(epochs=epochs, model=model)
        if cb_res is not None:
            status.update(cb_res)

    for epoch in range(1, epochs + 1):
        train_metrics.reset()
        model.train()

        for data in train_loader:
            sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)
            opt.zero_grad()
            logits = model(sample)

            loss = loss_fn(logits, label.float())
            loss.backward()
            opt.step()

            predictions = F.sigmoid(logits)
            train_metrics.update(predictions, label, loss)
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


def objective(trial: optuna.Trial):
    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    study_name = trial.study.study_name

    match study_name:
        case "MLP":
            model_layers = mlp.suggest_model(trial)
            model = nn.Sequential(*model_layers).to(DEVICE)
        case "CNN":
            model = cnn.suggest_model(trial)
        case _:
            raise ValueError(f"Study {study_name} is not implemented")

    oocha_dir = trial.study.user_attrs["oocha_dir"]
    epochs = trial.study.user_attrs["epochs"]
    dataset = trial.study.user_attrs["set"]
    samples = trial.study.user_attrs["samples"]

    run = {
        "experiment": study_name,
        "dataset": {
            "set": dataset,
            "samples": samples,
            "preprocessing": {},
            "test_set": {"augmentation": "random_shift"},
        },
    }
    run["params"] = trial.params
    run_hash = DeepHash(run)[run]

    running_bac = 0.0
    running_loss = 0.0
    running_f1 = 0.0
    splits = 5
    run_dir = RESULTS_DIR / study_name / run_hash
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(
        skfold(DATA_DIR / dataset, oocha_dir, batch_size, n_splits=splits)
    ):
        if fold_idx == 0:
            run["model_arch"] = str(model)
            run_dir.mkdir(parents=True, exist_ok=True)
            trial.set_user_attr("run_dir", str(run_dir))

        # Reset model between folds
        reset_module_weights(model)
        opt = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=tensor(train_loader.dataset.get_pos_weight()),
        )
        val_loss_fn = nn.BCEWithLogitsLoss()

        train_metrics = Metrics("train", len(train_loader.dataset), DEVICE, fold_idx)
        val_metrics = Metrics("val", len(val_loader.dataset), DEVICE, fold_idx)
        test_metrics = Metrics("test", len(test_loader.dataset), DEVICE, fold_idx)

        # Train model
        model_path = run_dir / f"{fold_idx}_model.pt"
        train_metrics, val_metrics, status = fit(
            model,
            opt,
            loss_fn,
            val_loss_fn,
            train_metrics,
            val_metrics,
            train_loader,
            val_loader,
            [
                EarlyStopping(patience=200),
                EpochInfoLogger(50),
                ModelCheckPoint(model_path),
            ],
            epochs,
        )

        running_loss += min(val_metrics.saved_metrics["loss"])
        running_bac += max(val_metrics.saved_metrics["bac"])
        running_f1 += max(val_metrics.saved_metrics["f1"])

        model.load_state_dict(torch.load(model_path))
        # Test the best model
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)
                logits = model(sample)

                loss = val_loss_fn(logits, label.float())

                predictions = F.sigmoid(logits)
                test_metrics.update(predictions, label, loss)
        test_metrics.save_metrics(0)
        train_metrics.write_metrics(run_dir)
        val_metrics.write_metrics(run_dir)
        test_metrics.write_metrics(run_dir)

        # End experiment
        run["results"] = {
            "best_train": train_metrics.best_metrics(),
            "best_val": val_metrics.best_metrics(),
        }

        # Save result
        with open(run_dir / f"{fold_idx}_results.json", "w") as f:
            json.dump(run, f, skipkeys=True, indent=4)

    return running_f1 / splits, running_bac / splits, running_loss / splits


def tune(study_name: str, n_trials: int, epochs: int, oocha_dir: str, timeout: float | None = None):
    logger = structlog.get_logger()
    storage = JournalStorage(JournalFileStorage(f"{RESULTS_DIR}/journal.log"))

    dataset = "clean_df.csv"

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        directions=["maximize", "maximize", "minimize"],
        load_if_exists=True,
    )
    study.set_user_attr("oocha_dir", oocha_dir)
    study.set_user_attr("epochs", epochs)
    study.set_user_attr("set", dataset)
    study.set_user_attr("samples", 1500)
    study.set_metric_names(["f1", "bac", "loss"])

    # Cache signals so first trial isn't overly long
    ds = TransitionDataset(pl.read_csv(DATA_DIR / dataset), oocha_dir, [CropSample(15)])
    dl = DataLoader(ds, 16)
    for i, _ in enumerate(dl):
        logger.info("Cached signals", cached=i * 16, total=len(ds))

    logger.info("Starting study", dataset=dataset, study_name=study_name)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        timeout=timeout,
    )
