import typer
import structlog
import sys
import os
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def tune_model(
    study_name: str,
    model_name: str,
    dataset_name: str,
    oocha_dir: str,
    n_splits: int = 5,
    n_trials: int = 200,
    epochs: int = 1000,
    timeout: Annotated[Optional[float], typer.Option(help="In seconds")] = None,
) -> None:
    from tune import tune

    tune(study_name, model_name, dataset_name, n_splits, n_trials, epochs, oocha_dir, timeout)


@app.command()
def check_gpu():
    import torch

    logger = structlog.get_logger()

    try:
        logger.info("GPU", env=os.environ["CUDA_VISIBLE_DEVICES"])
    except KeyError as e:
        logger.warning(f"Environment variable not set: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device type", device=device)

    if device.type == "cuda":
        logger.info("Device", device_name=torch.cuda.get_device_name(device))
        logger.info(
            f"Mem:\n Allocated: {round(torch.cuda.memory_allocated(device) / 1024 **3, 1)}GB\n Cached: {round(torch.cuda.memory_reserved(device)/1024**3, 1)}GB"
        )

    try:
        logger.info("Device capabilities", caps=torch.cuda.get_device_capability())
    except AssertionError as e:
        logger.warning(e)


@app.command()
def viewer():
    import viewer
    from plotly_resampler import register_plotly_resampler

    register_plotly_resampler()

    viewer.viewer()


@app.command()
def optuna_dashboard(
    journal_path: str = "results/journal.log",
    host="127.0.0.1",
    port=8080,
):
    from optuna_dashboard import run_server
    from optuna.storages import JournalFileStorage, JournalStorage
    from utils import project_dir

    storage = project_dir() / journal_path
    journal = JournalStorage(JournalFileStorage(str(storage)))
    run_server(journal, host=host, port=port)


def main():
    processors = structlog.get_config()["processors"]
    processors.pop()  # Remove the renderer
    if sys.stderr.isatty():
        processors += [structlog.dev.ConsoleRenderer()]
    else:
        processors += [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    structlog.configure()
    app()


if __name__ == "__main__":
    main()
