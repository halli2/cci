import json
from typing import Any, Dict

import numpy as np
import optuna
import torch
from dataset import skfold
from deepdiff import DeepHash
from metrics import Metrics
from models import mlp
from rich.live import Live
from rich.progress import (
    Progress,
    TextColumn,
    TimeElapsedColumn,
    track,
)
from rich.table import Table
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import project_dir

RESULTS_DIR = project_dir() / "results"
SAMPLE_LENGTH = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.BCEWithLogitsLoss,
    val_loss_fn: nn.BCEWithLogitsLoss,
    train_metrics: Metrics,
    val_metrics: Metrics,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
) -> tuple[Metrics, Metrics, Dict[str, Any]]:
    table = Table("Training model)
    metric_info = Progress(TextColumn("{task.description}"))
    task_metrics = metric_info.add_task("Metrics")
    progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
    task_epoch = progress.add_task("Epochs")
    task_train = progress.add_task("Train")
    task_validation = progress.add_task("Validation")
    table.add_row(progress)
    table.add_row(metric_info)

    best_loss = np.inf
    best_model = model.state_dict()
    with Live(table):
        for epoch in progress.track(range(1, epochs + 1), description="Epochs", task_id=task_epoch):
            progress.reset(task_validation)
            train_metrics.reset()
            model.train()

            for data in progress.track(train_loader, description="Training", task_id=task_train):
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
                for data in progress.track(val_loader, description="Validation", task_id=task_validation):
                    sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)
                    logits = model(sample)

                    loss = val_loss_fn(logits, label.float())

                    predictions = F.sigmoid(logits)
                    val_metrics.update(predictions, label, loss)
            val_metrics.save_metrics(epoch)

            validation_values = val_metrics.compute()
            training_values = train_metrics.compute()
            metric_info.update(
                task_id=task_metrics,
                description=f"\nTraining\n Acc:{training_values['acc']:.3f}\n Loss{training_values['loss']:.3f}\n"
                f"Validation\n Acc:{validation_values['acc']:.3f}\n Loss:{validation_values['loss']:.3f}\n",
            )
            if validation_values["loss"] < best_loss:
                best_loss = validation_values["loss"]
                best_model = model.state_dict()

    return train_metrics, val_metrics, best_model


def reset_module_weights(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        if hasattr(module, "children"):
            for child in module.children():
                reset_module_weights(child)


def objective(trial: optuna.Trial):
    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    model_layers = mlp.suggest_model(trial)
    model = nn.Sequential(*model_layers).to(DEVICE)

    study_name = trial.study.study_name
    oocha_dir = trial.study.user_attrs["oocha_dir"]

    run = {
        "experiment": study_name,
        "dataset": {
            "set": "data/clean_df.csv",
            "samples": 1500,
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
        skfold("data/clean_df.csv", oocha_dir, batch_size, n_splits=splits)
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
        train_metrics, val_metrics, best_model = fit(
            model,
            opt,
            loss_fn,
            val_loss_fn,
            train_metrics,
            val_metrics,
            train_loader,
            val_loader,
            epochs=100,
        )

        running_loss += min(val_metrics.saved_metrics["loss"])
        running_bac += max(val_metrics.saved_metrics["bac"])
        running_f1 += max(val_metrics.saved_metrics["f1"])

        model.load_state_dict(best_model)
        # Test the best model
        model.eval()
        with torch.no_grad():
            for data in track(test_loader, description="Testing"):
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
        torch.save(model.state_dict(), run_dir / f"{fold_idx}_model.pt")

    return running_f1 / splits, running_bac / splits, running_loss / splits


def tune(study_name: str, n_trials: int, oocha_dir: str):
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{RESULTS_DIR}/optuna.db",
        heartbeat_interval=10,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
    )

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        directions=["maximize", "maximize", "minimize"],
        load_if_exists=True,
    )
    study.set_user_attr("oocha_dir", oocha_dir)
    study.set_metric_names(["f1", "bac", "loss"])
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )
