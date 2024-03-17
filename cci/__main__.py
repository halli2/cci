import typer
from tune import tune
import structlog
import sys

app = typer.Typer()


@app.command()
def tune_mlp(study_name: str, n_trials: int, oocha_dir: str) -> None:
    tune(study_name, n_trials, oocha_dir)


@app.command()
def check_gpu():
    import os

    import torch

    logger = structlog.get_logger()

    try:
        logger.info("GPU", os.environ["CUDA_VISIBLE_DEVICES"])
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
    print(structlog.get_config())
    app()


if __name__ == "__main__":
    main()
