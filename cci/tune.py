import json

import optuna
import torch
from dataset import CropSample, TransitionDataset, skfold
from metrics import Metrics
from models import cnn, mlp
from models.layers import reset_module_weights
from optuna.storages import JournalFileStorage, JournalStorage
from torch import nn, optim, tensor
from torch.nn import functional as F
from utils import project_dir
from train import EarlyStopping, ModelCheckPoint, EpochInfoLogger, fit

RESULTS_DIR = project_dir() / "results"
DATA_DIR = project_dir() / "data"
SAMPLE_LENGTH = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial: optuna.Trial):
    study_name = trial.study.study_name
    model_name = trial.study.user_attrs["model"]

    match model_name:
        case "MLP":
            model_layers = mlp.suggest_model(trial)
            model = nn.Sequential(*model_layers).to(DEVICE)
        case "CNN":
            model = cnn.suggest_model(trial).to(DEVICE)
        case "CNN_bigger_kernel":
            model = cnn.suggest_model(trial, max_kernel_size=250).to(DEVICE)
        case "CNN_bigger_kernel_mlp":
            model = cnn.suggest_model(trial, max_kernel_size=250, mlp_layers=1).to(DEVICE)
        case "CNN_even_bigger_kernel_mlp":
            model = cnn.suggest_model(trial, max_kernel_size=1500, mlp_layers=1).to(DEVICE)
        case _:
            raise ValueError(f"Study {study_name} is not implemented")

    oocha_dir = trial.study.user_attrs["oocha_dir"]
    epochs = trial.study.user_attrs["epochs"]
    dataset = trial.study.user_attrs["dataset"]
    samples = trial.study.user_attrs["samples"]

    run = {
        "experiment": study_name,
        "dataset": {
            "set": dataset,
            "samples": samples,
            "preprocessing": {},
            # "test_set": {"augmentation": "random_shift"},
        },
    }
    run["params"] = trial.params

    running_bac = 0.0
    running_loss = 0.0
    running_f1 = 0.0
    splits = trial.study.user_attrs["splits"]
    run_dir = RESULTS_DIR / study_name / str(trial.number)

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(
        skfold(DATA_DIR / dataset, oocha_dir, batch_size=32, n_splits=splits)
    ):
        if fold_idx == 0:
            run["model_arch"] = str(model)
            run_dir.mkdir(parents=True, exist_ok=True)
            trial.set_user_attr("run_dir", str(run_dir))

        # Reset model between folds
        reset_module_weights(model)
        opt = optim.Adam(model.parameters(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=tensor(train_loader.dataset.get_pos_weight()),
        )

        train_metrics = Metrics("train", len(train_loader.dataset), DEVICE, fold_idx)
        val_metrics = Metrics("val", len(val_loader.dataset), DEVICE, fold_idx)
        test_metrics = Metrics("test", len(test_loader.dataset), DEVICE, fold_idx)

        # Train modeln_splits
        model_path = run_dir / f"{fold_idx}_model.pt"
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
                ModelCheckPoint(model_path),
            ],
            epochs,
        )

        running_loss += min(val_metrics.saved_metrics["loss"])
        running_bac += max(val_metrics.saved_metrics["bac"])
        running_f1 += max(val_metrics.saved_metrics["f1"])

        model = torch.jit.load(model_path)
        # model.load_state_dict(torch.load(model_path))
        # Test the best model
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                sample, label = data["signal"].to(DEVICE), data["label"].to(DEVICE)
                logits = model(sample)

                loss = loss_fn(logits, label.float())

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
            "test_metrics": test_metrics.saved_metrics,
        }

        # Save result
        with open(run_dir / f"{fold_idx}_results.json", "w") as f:
            json.dump(run, f, skipkeys=True, indent=4)

    # return running_f1 / splits, running_bac / splits, running_loss / splits
    return running_bac / splits


def tune(
    study_name: str,
    model_name: str,
    dataset_name: str,
    splits: int,
    n_trials: int,
    epochs: int,
    oocha_dir: str,
    timeout: float | None = None,
):
    # logger = structlog.get_logger()
    RESULTS_DIR.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileStorage(f"{RESULTS_DIR}/journal.log"))

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        directions=["maximize"],
        load_if_exists=True,
    )
    study.set_user_attr("oocha_dir", oocha_dir)
    study.set_user_attr("epochs", epochs)
    study.set_user_attr("samples", 1500)
    study.set_user_attr("model", model_name)
    study.set_user_attr("dataset", dataset_name)
    study.set_user_attr("splits", splits)
    # study.set_user_attr("model", )
    # study.set_metric_names(["f1", "bac", "loss"])
    study.set_metric_names(["bac"])

    # Cache signals so first trial isn't overly long
    # ds = TransitionDataset(pl.read_csv(DATA_DIR / dataset), oocha_dir, [CropSample(15)])
    # dl = DataLoader(ds, 16)
    # for i, _ in enumerate(dl):
    # logger.info("Cached signals", cached=(i+1) * 16, total=len(ds))

    # logger.info("Starting study", dataset=dataset, study_name=study_name)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        timeout=timeout,
    )
