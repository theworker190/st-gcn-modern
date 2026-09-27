#!/usr/bin/env python
"""Prepare already-extracted Kinetics-Skeleton JSON data for ST-GCN."""

import argparse
import json
import pickle
from pathlib import Path

from numpy.lib.format import open_memmap

from stgcn.data.kinetics import read_kinetics_sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skeleton-dir",
        required=True,
        type=Path,
        help="Directory containing kinetics_train/, kinetics_val/, and label JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/kinetics400")
    )
    parser.add_argument("--num-person-in", type=int, default=5)
    parser.add_argument("--num-person-out", type=int, default=2)
    return parser.parse_args()


def prepare_split(
    source_dir,
    label_path,
    output_dir,
    split,
    num_person_in,
    num_person_out,
):
    with label_path.open(encoding="utf-8") as label_file:
        label_info = json.load(label_file)
    sample_paths = [
        path
        for path in sorted(source_dir.glob("*.json"))
        if label_info[path.stem]["has_skeleton"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    data = open_memmap(
        output_dir / f"{split}_data.npy",
        dtype="float32",
        mode="w+",
        shape=(len(sample_paths), 3, 300, 18, num_person_out),
    )
    labels = []
    for index, path in enumerate(sample_paths):
        skeleton, label = read_kinetics_sample(
            path,
            max_frames=300,
            num_person_in=num_person_in,
            num_person_out=num_person_out,
        )
        expected_label = int(label_info[path.stem]["label_index"])
        if label != expected_label:
            raise ValueError(f"Label mismatch in {path}")
        data[index] = skeleton
        labels.append(label)
        if (index + 1) % 100 == 0 or index + 1 == len(sample_paths):
            print(f"{split}: {index + 1}/{len(sample_paths)}")
    data.flush()

    with (output_dir / f"{split}_label.pkl").open("wb") as label_file:
        pickle.dump(([path.name for path in sample_paths], labels), label_file)


def main():
    args = parse_args()
    for split in ("train", "val"):
        prepare_split(
            args.skeleton_dir / f"kinetics_{split}",
            args.skeleton_dir / f"kinetics_{split}_label.json",
            args.output_dir,
            split,
            args.num_person_in,
            args.num_person_out,
        )


if __name__ == "__main__":
    main()
