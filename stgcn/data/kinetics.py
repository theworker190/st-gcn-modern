"""Reading of already-extracted Kinetics-Skeleton JSON samples."""

import json
from pathlib import Path

import numpy as np

from .transforms import auto_pad


def read_kinetics_sample(
    path,
    *,
    max_frames=300,
    num_person_in=5,
    num_person_out=2,
):
    """Convert one upstream Kinetics-Skeleton JSON file to ``(C,T,V,M)``."""
    with Path(path).open(encoding="utf-8") as sample_file:
        sample = json.load(sample_file)

    data = np.zeros((3, max_frames, 18, num_person_in), dtype=np.float32)
    for frame in sample["data"]:
        frame_index = int(frame["frame_index"])
        if not 0 <= frame_index < max_frames:
            continue
        for person, skeleton in enumerate(frame["skeleton"][:num_person_in]):
            pose = skeleton["pose"]
            data[0, frame_index, :, person] = pose[0::2]
            data[1, frame_index, :, person] = pose[1::2]
            data[2, frame_index, :, person] = skeleton["score"]

    data[0:2] -= 0.5
    data[0][data[2] == 0] = 0
    data[1][data[2] == 0] = 0

    sort_index = (-data[2].sum(axis=1)).argsort(axis=1)
    for frame, order in enumerate(sort_index):
        data[:, frame, :, :] = data[:, frame, :, order].transpose((1, 2, 0))
    data = data[:, :, :, :num_person_out]
    return auto_pad(data, max_frames), int(sample["label_index"])
