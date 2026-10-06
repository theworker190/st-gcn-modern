"""Safe YAML loading and direct construction of configured objects."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from .data import SkeletonDataset
from .model import STGCN
from .normalization import configure_batch_norm


def load_config(path):
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    for key in ("model", "dataset", "training"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Configuration is missing the {key!r} mapping")
    return config


def build_model(config):
    model = STGCN(**dict(config["model"]))
    configure_batch_norm(model, config["training"].get("batch_norm_shards", 1))
    return model


def build_dataset(config, split):
    dataset_config = config["dataset"]
    if not isinstance(dataset_config, Mapping) or split not in dataset_config:
        raise ValueError(f"Dataset split {split!r} is not configured")
    split_config = dataset_config[split]
    if not isinstance(split_config, Mapping):
        raise ValueError(f"Dataset split {split!r} must be a mapping")
    return SkeletonDataset(**dict(split_config))
