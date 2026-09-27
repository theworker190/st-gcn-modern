"""Small checkpoint helpers, including original ST-GCN compatibility."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


def normalize_state_dict_keys(state_dict):
    """Remove the prefix added by ``torch.nn.DataParallel`` when present."""
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def extract_model_state(checkpoint):
    """Accept an upstream raw state dict or a modern training checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")

    candidate = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break
    if not isinstance(candidate, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("Could not find a tensor model state dict in checkpoint")
    return normalize_state_dict_keys(candidate)


def load_checkpoint_file(path):
    """Load tensor-only checkpoint content on CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_model_weights(
    model,
    path,
    *,
    strict=True,
    ignore_prefixes=(),
):
    checkpoint = load_checkpoint_file(path)
    state_dict = extract_model_state(checkpoint)
    if ignore_prefixes:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(ignore_prefixes)
        }
        strict = False
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    epoch,
    config,
):
    state_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": dict(config),
        },
        Path(path),
    )


def restore_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
):
    """Restore training state and return the next epoch index."""
    checkpoint = load_checkpoint_file(path)
    model.load_state_dict(extract_model_state(checkpoint))
    if not isinstance(checkpoint, Mapping):
        return 0
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if isinstance(optimizer_state, Mapping):
        optimizer.load_state_dict(dict(optimizer_state))
    if isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(dict(scheduler_state))
    return int(checkpoint.get("epoch", 0))
