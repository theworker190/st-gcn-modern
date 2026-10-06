# ST-GCN Modern

A lightweight, unofficial modernization of [the original ST-GCN implementation](https://github.com/yysijie/st-gcn) by Sijie Yan, Yuanjun Xiong, and Dahua Lin ([AAAI 2018 paper](https://arxiv.org/abs/1801.07455)). For researchers who want the ST-GCN baseline without a larger toolbox: PyTorch, NumPy, and PyYAML, with simple training and evaluation scripts.

Supports NTU RGB+D 60 cross-subject, cross-view, and Kinetics-Skeleton 400. Inputs are already-extracted skeleton coordinates; pose estimation, video demos, and the legacy framework have been removed.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

Linux / macOS:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (with Python and pip installed):

```sh
pip install uv
```

Restart your terminal, then clone and install the project (Python 3.10+):

```sh
git clone https://github.com/theworker190/st-gcn-modern.git
cd st-gcn-modern
uv sync
```

## Prepare data

NTU: use the dataset's `.skeleton` files. The converter creates both benchmark splits and excludes the supplied missing-skeleton list.

```sh
uv run python tools/prepare_ntu.py --skeleton-dir /path/to/nturgb+d_skeletons
```

Kinetics: use the extracted skeleton release containing `kinetics_train/`, `kinetics_val/`, and their label JSON files.

```sh
uv run python tools/prepare_kinetics.py --skeleton-dir /path/to/kinetics-skeleton
```

Outputs:

```text
data/ntu60/xsub/
data/ntu60/xview/
data/kinetics400/
```

Each directory contains `train_data.npy`, `train_label.pkl`, `val_data.npy`, and `val_label.pkl`. Arrays use `(N, C, T, V, M)`: samples, channels, frames, joints, people. Label pickles contain `(sample_names, labels)`.

## Train

```sh
uv run python train.py --config configs/ntu60_xsub.yaml
```

Other configs: `configs/ntu60_xview.yaml` and `configs/kinetics400.yaml`. Checkpoints, config, and logs go to `runs/<benchmark>/`.

Use `--device cuda:0` or `--device cpu` to choose a device, `--output-dir runs/my_run` for a separate run, and `--resume runs/ntu60_xsub/epoch_010.pt` to resume.

**BatchNorm fidelity:** configs use `batch_norm_shards: 4` to preserve the original four-GPU normalization groups on one GPU. Keep the configured batch size for the reference setup. Batch size must be divisible by the shard count; changing the shard count requires fresh training.

## Evaluate

```sh
uv run python eval.py --config configs/ntu60_xsub.yaml --checkpoint runs/ntu60_xsub/epoch_080.pt
```

Reports loss, top-1, and top-5 accuracy. Original ST-GCN checkpoints are supported, including `module.`-prefixed weights; use the matching benchmark config. Add `--save-scores results/scores.pkl` to export predictions.

## Code and tests

Model, blocks, and graphs: `stgcn/model.py`, `stgcn/layers.py`, `stgcn/graph.py`. Data loading and transforms: `stgcn/data/`. Dataset converters: `tools/`.

```sh
uv sync --extra dev
uv run pytest
```

## Citation

```bibtex
@inproceedings{stgcn2018aaai,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}
```

Derived from the original ST-GCN repository under the [BSD-2-Clause license](LICENSE); upstream copyright notices are preserved.
