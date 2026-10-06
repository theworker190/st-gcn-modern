#!/usr/bin/env python
"""Train the original ST-GCN baseline from a small YAML configuration."""

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from stgcn.checkpoint import (
    load_checkpoint_file,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from stgcn.config import build_dataset, build_model, load_config
from stgcn.model import initialize_weights
from stgcn.runtime import configure_logging, evaluate, resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-GCN")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", help="Override the configured device, e.g. cpu or cuda:0")
    parser.add_argument("--output-dir", type=Path, help="Override the configured output directory")
    parser.add_argument("--resume", type=Path, help="Resume a modern training checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    output_dir = args.output_dir or Path(config.get("output_dir", "runs/stgcn"))
    device = resolve_device(args.device or str(config.get("device", "auto")))
    shards = training.get("batch_norm_shards", 1)
    if (
        not isinstance(shards, int) or isinstance(shards, bool)
        or shards < 1 or int(training["batch_size"]) % shards
    ):
        raise ValueError("batch_size must be divisible by positive batch_norm_shards")
    saved = load_checkpoint_file(args.resume) if args.resume else None
    if saved is not None:
        saved_shards = saved.get("config", {}).get("training", {}).get("batch_norm_shards", 1)
        if saved_shards != shards:
            raise ValueError("Resume checkpoint uses different batch_norm_shards; start a fresh run")
    seed_everything(int(config.get("seed", 1)))
    logger = configure_logging(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    workers = int(config.get("num_workers", 4))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    model = build_model(config)
    model.apply(initialize_weights)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["base_lr"]),
        momentum=float(training.get("momentum", 0.9)),
        nesterov=bool(training.get("nesterov", True)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(training["steps"]),
        gamma=0.1,
    )
    start_epoch = 0
    if args.resume:
        start_epoch = restore_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logger.info("Resumed %s at epoch %d", args.resume, start_epoch)

    epochs = int(training["epochs"])
    log_interval = int(training.get("log_interval", 100))
    eval_interval = int(training.get("eval_interval", 5))
    save_interval = int(training.get("save_interval", 10))
    logger.info("Training on %s with %d training samples", device, len(train_dataset))

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for iteration, (data, labels) in enumerate(train_loader):
            data = data.float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            scores = model(data)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            if iteration % log_interval == 0:
                logger.info(
                    "epoch %d/%d iteration %d loss %.4f lr %.6f",
                    epoch + 1,
                    epochs,
                    iteration,
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
        if total_samples == 0:
            raise ValueError("Training dataset is smaller than one full batch")
        logger.info("epoch %d mean training loss %.4f", epoch + 1, total_loss / total_samples)

        should_evaluate = (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs
        if should_evaluate:
            val_loss, accuracy, _ = evaluate(model, val_loader, device, criterion)
            logger.info(
                "epoch %d validation loss %.4f top-1 %.2f%% top-5 %.2f%%",
                epoch + 1,
                val_loss,
                accuracy[1] * 100,
                accuracy[5] * 100,
            )

        scheduler.step()
        should_save = (epoch + 1) % save_interval == 0 or epoch + 1 == epochs
        if should_save:
            checkpoint_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                config=config,
            )
            logger.info("Saved %s", checkpoint_path)


if __name__ == "__main__":
    main()
