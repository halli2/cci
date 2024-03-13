from typing import Any

import torch

# TODO: Add drift? Normalize? Baseline? Noise?


class RandomSample(object):
    """Sample randomly from the signal
    Arguments:
        output_size: length of the signal"""

    def __init__(self, output_size: int):
        self.output_size = output_size

    def __call__(self, sample) -> dict[str, Any]:
        signal, label = sample["signal"], sample["label"]
        new_start = torch.randint(0, len(signal) - self.output_size, (1,))

        sample = signal[new_start : new_start + self.output_size]
        return {"signal": sample, "label": label}


class ToTensor(object):
    """Convert to tensors"""

    def __call__(self, sample) -> dict[str, torch.Tensor]:
        signal, label = sample["signal"], sample["label"]
        return {"signal": torch.from_numpy(signal), "label": torch.tensor(label)}


class CropSample(object):
    """Crop sample to the last x sample points"""

    def __init__(self, output_size: int):
        self.output_size = output_size

    def __call__(self, sample) -> dict[str, Any]:
        signal, label = sample["signal"], sample["label"]

        sample = signal[-self.output_size :]
        return {"signal": sample, "label": label}
