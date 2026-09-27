#!/usr/bin/env python
"""Prepare NTU RGB+D 60 skeleton files for ST-GCN."""

import argparse
import pickle
import re
from pathlib import Path

from numpy.lib.format import open_memmap

from stgcn.data.ntu import read_xyz


TRAINING_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
}
TRAINING_CAMERAS = {2, 3}
NAME_PATTERN = re.compile(r"S\d{3}C(?P<camera>\d{3})P(?P<subject>\d{3})R\d{3}A(?P<action>\d{3})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ntu60"))
    parser.add_argument(
        "--ignored-samples",
        type=Path,
        default=Path("stgcn/data/ntu60_missing_skeletons.txt"),
    )
    parser.add_argument(
        "--protocol", choices=("all", "xsub", "xview"), default="all"
    )
    return parser.parse_args()


def collect_samples(
    skeleton_dir,
    ignored_samples,
    protocol,
    split,
):
    paths = []
    labels = []
    for path in sorted(skeleton_dir.glob("*.skeleton")):
        if path.name in ignored_samples:
            continue
        match = NAME_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        camera = int(match.group("camera"))
        subject = int(match.group("subject"))
        is_training = (
            camera in TRAINING_CAMERAS
            if protocol == "xview"
            else subject in TRAINING_SUBJECTS
        )
        if (split == "train") != is_training:
            continue
        paths.append(path)
        labels.append(int(match.group("action")) - 1)
    return paths, labels


def prepare_split(
    skeleton_dir,
    output_dir,
    ignored_samples,
    protocol,
    split,
):
    paths, labels = collect_samples(
        skeleton_dir, ignored_samples, protocol, split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split}_label.pkl").open("wb") as label_file:
        pickle.dump(([path.name for path in paths], labels), label_file)

    data = open_memmap(
        output_dir / f"{split}_data.npy",
        dtype="float32",
        mode="w+",
        shape=(len(paths), 3, 300, 25, 2),
    )
    for index, path in enumerate(paths):
        skeleton = read_xyz(path, max_body=2, num_joint=25)
        frames = min(skeleton.shape[1], 300)
        data[index, :, :frames, :, :] = skeleton[:, :frames, :, :]
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"{protocol}/{split}: {index + 1}/{len(paths)}")
    data.flush()


def main():
    args = parse_args()
    ignored_samples = {
        f"{line.strip()}.skeleton"
        for line in args.ignored_samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    protocols = ("xsub", "xview") if args.protocol == "all" else (args.protocol,)
    for protocol in protocols:
        for split in ("train", "val"):
            prepare_split(
                args.skeleton_dir,
                args.output_dir / protocol,
                ignored_samples,
                protocol,
                split,
            )


if __name__ == "__main__":
    main()
