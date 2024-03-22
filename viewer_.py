# %%

import os
from pathlib import Path

import polars as pl
import rerun as rr
import scipy

from cci.utils import project_dir

# http://vrl.cs.brown.edu/color

color = {
    "AS": [82, 239, 153],
    "VT": [17, 94, 65],
    "VF": [148, 210, 207],
    "PE": [44, 69, 125],
    "PR": [209, 121, 248],
    "as": [153, 28, 100],
    "vf": [172, 130, 180],
    "vt": [102, 51, 180],
    "pe": [238, 200, 241],
    "pr": [41, 140, 192],
    "CAS": [183, 209, 101],
    "CVT": [58, 166, 9],
    "CVF": [44, 245, 43],
    "CPE": [93, 64, 48],
    "CPR": [254, 183, 134],
    "cas": [177, 75, 50],
    "cvt": [251, 45, 76],
    "cvf": [251, 189, 19],
    "cpe": [116, 141, 19],
    "cpr": [37, 128, 254],
    "un": [246, 18, 168],
    "cun": [194, 24, 241],
    "dfb": [160, 127, 61],
}


rr.init("Signal Viewer")
rr.connect()

# %%
oocha_dir = Path(os.environ["OOCHA_DIR"])
clean_df = pl.read_csv(project_dir() / "data/clean_df.csv").with_row_index()
usefull_files = clean_df.select(pl.col("files")).to_series().unique()
original_df = pl.read_csv(project_dir() / "data/original.csv").filter(pl.col("files").is_in(usefull_files))

df = original_df.select(["files", "EPI", "EPI_1", "SMP_start", "SMP_stop", "SMP_stop_1"])
file, epi, epi_1, start, transition, stop = df.row(1)

signal = scipy.io.loadmat(
    oocha_dir / f"{file}.mat",
    simplify_cells=True,
)["SIGNALS"]["ecg_diff"]


# %%
tick = 1
for i, (epi, start, stop) in enumerate(
    df.filter(pl.col("files") == file).select(["EPI", "SMP_start", "SMP_stop"]).rows()
):
    # Check if used in training dataset
    try:
        clean_idx = clean_df.filter(pl.col("files") == file, pl.col("SMP_start") == start).select("index").item()
        print(clean_idx)
        log_name = f"ecg/{clean_idx}/{epi}/{i}"
    except ValueError as e:
        log_name = f"ecg/none/{epi}/{i}"
    rr.log(log_name, rr.SeriesLine(color=color[epi], name=epi), timeless=True)
    # Label

    for value in signal[start:stop]:
        rr.set_time_sequence("step", tick)
        rr.log(log_name, rr.Scalar(value))
        tick += 1

# %%
res_dir = project_dir() / "results_go/CNN/f2ce86739c5335e9cfdfe876be3515dc5a254c3d8848935ef354c0d5b074ef66"
# rr.log(rr.TextLog())

for i in range(5):
    train_df = pl.read_csv(res_dir / f"{i}_train.csv")
    for epoch, row in train_df.rows_by_key(key=["epoch"], named=True, unique=True).items():
        rr.set_time_sequence("epoch", epoch)
        for metric, value in row.items():
            rr.log(f"{metric}/train", rr.Scalar(value))
