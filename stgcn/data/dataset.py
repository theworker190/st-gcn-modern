"""Dataset for preprocessed ``(N, C, T, V, M)`` skeleton arrays."""

import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .transforms import auto_pad, random_choose, random_move, random_shift


class SkeletonDataset(Dataset):
    """Load prepared ST-GCN arrays and their upstream-compatible labels."""

    def __init__(
        self,
        data_path,
        label_path,
        *,
        random_choose_sequence=False,
        random_shift_sequence=False,
        random_move_sequence=False,
        window_size=-1,
        debug=False,
        mmap=True,
    ):
        with Path(label_path).open("rb") as label_file:
            sample_names, labels = pickle.load(label_file)
        data = np.load(data_path, mmap_mode="r" if mmap else None)

        limit = 100 if debug else None
        self.sample_names = list(sample_names[:limit])
        self.labels = np.asarray(labels[:limit], dtype=np.int64)
        self.data = data[:limit]
        if self.data.ndim != 5:
            raise ValueError(
                f"Expected data with shape (N, C, T, V, M), got {self.data.shape}"
            )
        if len(self.data) != len(self.labels):
            raise ValueError("Data and label files contain different sample counts")
        if random_choose_sequence and window_size < 1:
            raise ValueError("window_size must be positive when random choosing is enabled")

        self.random_choose_sequence = random_choose_sequence
        self.random_shift_sequence = random_shift_sequence
        self.random_move_sequence = random_move_sequence
        self.window_size = window_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Copy because augmentation is in-place and source arrays may be memory-mapped.
        data = np.array(self.data[index], dtype=np.float32, copy=True)
        if self.random_shift_sequence:
            data = random_shift(data)
        if self.random_choose_sequence:
            data = random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = auto_pad(data, self.window_size)
        if self.random_move_sequence:
            data = random_move(data)
        return data, int(self.labels[index])

    def top_k(self, scores, k):
        rank = scores.argsort(axis=1)
        hits = [label in rank[index, -k:] for index, label in enumerate(self.labels)]
        return float(np.mean(hits))
