import os
from plotly import express as px
from utils import project_dir
import polars as pl
from pathlib import Path
import scipy


def viewer():
    oocha_dir = Path(os.environ["OOCHA_DIR"])
    df = pl.read_csv(project_dir() / "data/clean_df.csv")
    for file, epi, epi_1, start, transition, stop in df.select(
        ["files", "EPI", "EPI_1", "SMP_start", "SMP_start_1", "SMP_stop_1"]
    ).rows():
        signal = scipy.io.loadmat(
            oocha_dir / f"{file}.mat",
            simplify_cells=True,
        )["SIGNALS"]["ecg_diff"]
        # sig_df =
        fig = px.line(signal[start:stop])
        fig.show()
        break
