"""Reader for the NTU RGB+D skeleton text format."""

from pathlib import Path

import numpy as np


def read_xyz(
    path, *, max_body=2, num_joint=25
):
    with Path(path).open(encoding="utf-8") as skeleton_file:
        num_frames = int(skeleton_file.readline())
        data = np.zeros((3, num_frames, num_joint, max_body), dtype=np.float32)
        for frame_index in range(num_frames):
            num_bodies = int(skeleton_file.readline())
            for body_index in range(num_bodies):
                skeleton_file.readline()  # Body metadata is not used by ST-GCN.
                body_joint_count = int(skeleton_file.readline())
                for joint_index in range(body_joint_count):
                    joint = skeleton_file.readline().split()
                    if body_index < max_body and joint_index < num_joint:
                        data[:, frame_index, joint_index, body_index] = tuple(
                            map(float, joint[:3])
                        )
    return data
