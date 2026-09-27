"""NumPy preprocessing retained from the original ST-GCN feeders."""

import random

import numpy as np


def auto_pad(
    data, size, random_pad=False
):
    channels, frames, joints, people = data.shape
    if frames >= size:
        return data
    begin = random.randint(0, size - frames) if random_pad else 0
    padded = np.zeros((channels, size, joints, people), dtype=data.dtype)
    padded[:, begin : begin + frames, :, :] = data
    return padded


def random_choose(
    data, size, auto_pad_short_sequences=True
):
    frames = data.shape[1]
    if frames == size:
        return data
    if frames < size:
        return auto_pad(data, size, random_pad=True) if auto_pad_short_sequences else data
    begin = random.randint(0, frames - size)
    return data[:, begin : begin + size, :, :]


def random_shift(data):
    frames = data.shape[1]
    shifted = np.zeros_like(data)
    valid_frame = (data != 0).sum(axis=(0, 2, 3)) > 0
    if not valid_frame.any():
        return shifted
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    size = end - begin
    bias = random.randint(0, frames - size)
    shifted[:, bias : bias + size, :, :] = data[:, begin:end, :, :]
    return shifted


def random_move(
    data,
    angle_candidate=(-10.0, -5.0, 0.0, 5.0, 10.0),
    scale_candidate=(0.9, 1.0, 1.1),
    transform_candidate=(-0.2, -0.1, 0.0, 0.1, 0.2),
    move_time_candidate=(1,),
):
    _, frames, joints, people = data.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, frames, frames / move_time).round().astype(int)
    node = np.append(node, frames)
    num_node = len(node)

    angles = np.random.choice(angle_candidate, num_node)
    scales = np.random.choice(scale_candidate, num_node)
    x_offsets = np.random.choice(transform_candidate, num_node)
    y_offsets = np.random.choice(transform_candidate, num_node)
    angle = np.zeros(frames)
    scale = np.zeros(frames)
    x_offset = np.zeros(frames)
    y_offset = np.zeros(frames)

    for index in range(num_node - 1):
        frame_slice = slice(node[index], node[index + 1])
        count = node[index + 1] - node[index]
        angle[frame_slice] = np.linspace(angles[index], angles[index + 1], count)
        scale[frame_slice] = np.linspace(scales[index], scales[index + 1], count)
        x_offset[frame_slice] = np.linspace(
            x_offsets[index], x_offsets[index + 1], count
        )
        y_offset[frame_slice] = np.linspace(
            y_offsets[index], y_offsets[index + 1], count
        )

    angle *= np.pi / 180
    theta = np.array(
        [
            [np.cos(angle) * scale, -np.sin(angle) * scale],
            [np.sin(angle) * scale, np.cos(angle) * scale],
        ]
    )
    for frame in range(frames):
        xy = data[0:2, frame, :, :]
        transformed = theta[:, :, frame] @ xy.reshape(2, -1)
        transformed[0] += x_offset[frame]
        transformed[1] += y_offset[frame]
        data[0:2, frame, :, :] = transformed.reshape(2, joints, people)
    return data


