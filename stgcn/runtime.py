"""Focused runtime helpers shared by the train and evaluation scripts."""

import logging
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def configure_logging(output_dir=None):
    logger = logging.getLogger("stgcn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def top_k_accuracy(scores, labels, k):
    effective_k = min(k, scores.size(1))
    predictions = scores.topk(effective_k, dim=1).indices
    return float(predictions.eq(labels.view(-1, 1)).any(dim=1).float().mean().item())


def evaluate(
    model,
    loader,
    device,
    criterion,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    score_batches = []
    label_batches = []
    with torch.no_grad():
        for data, labels in loader:
            data = data.float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            scores = model(data)
            batch_size = labels.size(0)
            total_loss += float(criterion(scores, labels).item()) * batch_size
            total_samples += batch_size
            score_batches.append(scores.cpu())
            label_batches.append(labels.cpu())

    if total_samples == 0:
        raise ValueError("Evaluation dataset is empty")
    all_scores = torch.cat(score_batches)
    all_labels = torch.cat(label_batches)
    accuracy = {k: top_k_accuracy(all_scores, all_labels, k) for k in (1, 5)}
    return total_loss / total_samples, accuracy, all_scores.numpy()
