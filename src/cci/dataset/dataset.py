from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import polars as pl
import scipy
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """ECG Transition dataset"""

    def __init__(
        self,
        df: pl.DataFrame,
        root_dir: str,
        transforms: Optional[List[Callable]] = None,
        cache: Optional[dict[str, Any]] = None,
    ):
        """Arguments:
        csv_file: Path to csv file with annotations
        root_dir: Path to data folder
        transform: Optional transforms to be applied on a sample."""
        self.df = df
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.cache = cache

    def get_pos_weight(self):
        ds_size = len(self.df)
        pos_size = len(self.df.filter(pl.col("Class Label") == 1))

        return (ds_size - pos_size) / pos_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.to_list()

        filename, label, start, stop = self.df.select(
            ["files", "Class Label", "SMP_start", "SMP_stop"],
        ).row(index)

        if self.cache:
            key = f"{filename} {start} {stop}"
            signal = self.cache[key]
        else:
            signal = get_signal(self.root_dir / f"{filename}.mat", start, stop)

        sample = {"signal": signal, "label": label}

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample


def get_signal(signal_path: Path, start: int, stop: int):
    return scipy.io.loadmat(
        signal_path,
        simplify_cells=True,
    )["SIGNALS"]["ecg_diff"].astype(np.float32)[start:stop]


def create_cache(df: pl.DataFrame, root_dir: Path) -> dict[str, Any]:
    cache = {}
    for filename, start, stop in df.select(
        ["files", "SMP_start", "SMP_stop"],
    ).iter_rows():
        signal_path = root_dir / f"{filename}.mat"
        key = f"{filename} {start} {stop}"
        cache[key] = get_signal(signal_path, start, stop)
    return cache
