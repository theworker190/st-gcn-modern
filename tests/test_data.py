import pickle
import random

import numpy as np

from stgcn.data import SkeletonDataset
from stgcn.data.transforms import auto_pad, random_choose, random_shift


def test_dataset_preserves_ctvm_shape_and_labels(tmp_path):
    data_path = tmp_path / "data.npy"
    label_path = tmp_path / "label.pkl"
    np.save(data_path, np.ones((2, 3, 8, 25, 2), dtype=np.float32))
    with label_path.open("wb") as label_file:
        pickle.dump((["sample-a", "sample-b"], [4, 9]), label_file)

    dataset = SkeletonDataset(data_path, label_path, window_size=12)
    sample, label = dataset[1]

    assert sample.shape == (3, 12, 25, 2)
    assert sample.dtype == np.float32
    assert label == 9
    np.testing.assert_array_equal(sample[:, :8], 1)
    np.testing.assert_array_equal(sample[:, 8:], 0)


def test_temporal_transforms_have_upstream_shapes():
    data = np.zeros((3, 10, 18, 2), dtype=np.float32)
    data[:, 2:6] = 1

    assert auto_pad(data, 14).shape == (3, 14, 18, 2)
    random.seed(3)
    assert random_choose(data, 6).shape == (3, 6, 18, 2)
    random.seed(3)
    shifted = random_shift(data)
    assert shifted.shape == data.shape
    assert np.count_nonzero(shifted) == np.count_nonzero(data)
