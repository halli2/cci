import functools
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

import numpy as np
import polars as pl
import scipy
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import project_dir

from .transforms import CropSample, RandomSample, ToTensor

# Use fixed seed to always get same test set
RANDOM_STATE = 0


class TransitionDataset(Dataset):
    """ECG Transition dataset"""

    def __init__(
        self,
        df: pl.DataFrame,
        root_dir: str | Path,
        transforms: Optional[List[Callable]] = None,
    ):
        """Arguments:
        df: Dataframe
        root_dir: Path to data folder
        transform: Optional transforms to be applied on a sample."""
        self.df = df
        self.root_dir = Path(root_dir)
        self.transforms = transforms

    def get_pos_weight(self) -> float:
        ds_size = len(self.df)
        pos_size = len(self.df.filter(pl.col("Class") == 1))

        return (ds_size - pos_size) / pos_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.to_list()

        filename, label, start, stop = self.df.select(
            ["file", "Class", "Start", "Stop"],
        ).row(index)

        signal = get_signal(self.root_dir / f"{filename}.mat", start, stop)
        sample = {"signal": signal, "label": label}

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample


def split_train_test(csv: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns train_val_df and test_df. Uses a fixed seed to always get the same test set"""
    df = pl.read_csv(csv).with_row_index()
    labels_series = df.select("Class").to_series()
    labels = labels_series.to_numpy()

    train_val_idx, test_idx = train_test_split(
        range(len(df)),
        stratify=labels,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    train_val_df = df.filter(pl.col("index").is_in(train_val_idx))
    test_df = df.filter(pl.col("index").is_in(test_idx))

    # Reindex
    train_val_df = train_val_df.drop("index").with_row_index()
    test_df = test_df.drop("index").with_row_index()
    return train_val_df, test_df


def skfold(
    csv: str | Path,
    root_dir: str | Path,
    batch_size: int,
    sample_length=1500,
    n_splits=5,
    shuffle=True,
    random_state: int = RANDOM_STATE,
) -> Generator[
    tuple[DataLoader[TransitionDataset], DataLoader[TransitionDataset], DataLoader[TransitionDataset]], None, None
]:
    """Generator for Straitifed K Fold"""
    dataset_folder = project_dir() / "data"
    test_df = pl.read_csv(dataset_folder / "test.csv")
    test_dataset = TransitionDataset(
        test_df,
        root_dir,
        transforms=[
            CropSample(sample_length),
            ToTensor(),
        ],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )
    for i in range(n_splits):
        train_df = pl.read_csv(dataset_folder / f"train_{i}.csv")
        val_df = pl.read_csv(dataset_folder / f"val_{i}.csv")

        train_dataset = TransitionDataset(
            train_df,
            root_dir,
            transforms=[
                RandomSample(sample_length),
                ToTensor(),
            ],
        )
        val_dataset = TransitionDataset(
            val_df,
            root_dir,
            transforms=[
                CropSample(sample_length),
                ToTensor(),
            ],
        )

        test_dataset = TransitionDataset(
            test_df,
            root_dir,
            transforms=[
                CropSample(sample_length),
                ToTensor(),
            ],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        yield train_loader, val_loader, test_loader

    # train_val_df, test_df = split_train_test(csv)
    # labels_series = train_val_df.select("Class Label").to_series()
    # labels = labels_series.to_numpy()
    # fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    # for train_idx, val_idx in fold.split(np.zeros(len(train_val_df)), labels):
    # train_df = train_val_df.filter(pl.col("index").is_in(train_idx))
    # val_df = train_val_df.filter(pl.col("index").is_in(val_idx))
    # train_dataset = TransitionDataset(
    #     train_df,
    #     root_dir,
    #     transforms=[
    #         RandomSample(sample_length),
    #         ToTensor(),
    #     ],
    # )
    # val_dataset = TransitionDataset(
    #     val_df,
    #     root_dir,
    #     transforms=[
    #         CropSample(sample_length),
    #         ToTensor(),
    #     ],
    # )
    # test_dataset = TransitionDataset(
    #     test_df,
    #     root_dir,
    #     transforms=[
    #         CropSample(sample_length),
    #         ToTensor(),
    #     ],
    # )
    # test_dataset = TransitionDataset(
    #     test_df,
    #     root_dir,
    #     transforms=[
    #         CropSample(sample_length),
    #         ToTensor(),
    #     ],
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    # )

    # yield train_loader, val_loader, test_loader


@functools.lru_cache(maxsize=None)
def get_signal(signal_path: Path, start: int, stop: int):
    return scipy.io.loadmat(
        signal_path,
        simplify_cells=True,
    )["SIGNALS"]["ecg_diff"].astype(np.float32)[start:stop]
