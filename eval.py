#!/usr/bin/env python
"""Evaluate an ST-GCN checkpoint."""

import argparse
import pickle
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from stgcn.checkpoint import load_model_weights
from stgcn.config import build_dataset, build_model, load_config
from stgcn.runtime import configure_logging, evaluate, resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", help="Override the configured device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size")
    parser.add_argument("--save-scores", type=Path, help="Write per-sample score vectors")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    device = resolve_device(args.device or str(config.get("device", "auto")))
    seed_everything(int(config.get("seed", 1)))
    logger = configure_logging()

    dataset = build_dataset(config, "val")
    workers = int(config.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(training.get("eval_batch_size", 64)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_model(config).to(device)
    load_model_weights(model, args.checkpoint)
    loss, accuracy, scores = evaluate(
        model, loader, device, nn.CrossEntropyLoss()
    )
    logger.info("loss %.4f top-1 %.2f%% top-5 %.2f%%", loss, accuracy[1] * 100, accuracy[5] * 100)

    if args.save_scores:
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        result = dict(zip(dataset.sample_names, scores, strict=True))
        with args.save_scores.open("wb") as score_file:
            pickle.dump(result, score_file)
        logger.info("Saved scores to %s", args.save_scores)


if __name__ == "__main__":
    main()
