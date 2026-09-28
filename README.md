# ST-GCN: a modern, faithful baseline

This repository is an unofficial, lightweight modernization of the original
[`yysijie/st-gcn`](https://github.com/yysijie/st-gcn) implementation of
**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition** (AAAI 2018). It is intentionally focused on one job: training and
evaluating the original ST-GCN algorithm on already-extracted skeleton
coordinates.

The model topology, graph layouts and partitioning, `(C, T, V, M)` data layout,
augmentation behavior, weight initialization, optimizer defaults, learning-rate
milestones, and canonical benchmark protocols follow the upstream code. The
surrounding Python has been reduced to a normal package and two direct PyTorch
entry points. This fork does not add newer GCN variants or claim improved
accuracy.

## Supported benchmarks

- NTU RGB+D 60 cross-subject (`configs/ntu60_xsub.yaml`)
- NTU RGB+D 60 cross-view (`configs/ntu60_xview.yaml`)
- Kinetics-Skeleton 400 (`configs/kinetics400.yaml`)

Each prepared array has shape `(N, C, T, V, M)`, where `N` is samples, `C` is
coordinate/confidence channels, `T` is frames, `V` is joints, and `M` is people.
NTU uses 3D `(x, y, z)` coordinates with 25 joints; Kinetics-Skeleton uses
`(x, y, confidence)` with 18 joints.

## Setup with uv

Python 3.10 or newer is required. From the repository root:

```console
uv sync
```

For the test dependency as well:

```console
uv sync --extra dev
```

`uv` creates and manages the environment; no separate installation of an
internal framework is needed. Hardware-specific PyTorch installations can be
selected using uv's documented package-index configuration before syncing.

## Prepare data

The repository never converts RGB video to poses. Obtain the skeleton modality
from the dataset owner or the already-extracted Kinetics-Skeleton release, then
convert those coordinates to the compact arrays used during training.

### NTU RGB+D 60

Download the NTU RGB+D skeleton files and run:

```console
uv run python tools/prepare_ntu.py --skeleton-dir /path/to/nturgb+d_skeletons
```

This creates both protocols under `data/ntu60/xsub/` and
`data/ntu60/xview/`. The official missing-skeleton list is included at
`stgcn/data/ntu60_missing_skeletons.txt`. Use `--protocol xsub` or
`--protocol xview` to prepare only one split.

### Kinetics-Skeleton 400

Point the converter at the upstream extracted-skeleton directory containing
`kinetics_train/`, `kinetics_val/`, and their label JSON files:

```console
uv run python tools/prepare_kinetics.py --skeleton-dir /path/to/kinetics-skeleton
```

The output is written to `data/kinetics400/`. The 400 label names are retained
at `stgcn/data/kinetics400_labels.txt` for reference. Pose estimation and raw
video decoding are deliberately not part of this repository.

Prepared files follow this layout:

```text
data/
├── ntu60/
│   ├── xsub/{train,val}_{data.npy,label.pkl}
│   └── xview/{train,val}_{data.npy,label.pkl}
└── kinetics400/{train,val}_{data.npy,label.pkl}
```

Each label pickle contains `(sample_names, integer_labels)`, matching the
historical release format.

## Train

Choose one canonical config:

```console
uv run python train.py --config configs/ntu60_xsub.yaml
uv run python train.py --config configs/ntu60_xview.yaml
uv run python train.py --config configs/kinetics400.yaml
```

Checkpoints, the resolved config, and a plain-text log are written below the
config's `output_dir`. Use `--device cpu`, `--device cuda:0`, or another PyTorch
device string to override automatic device selection. Resume a checkpoint with
`--resume runs/ntu60_xsub/epoch_010.pt`.

## Evaluate

```console
uv run python eval.py \
  --config configs/ntu60_xsub.yaml \
  --checkpoint runs/ntu60_xsub/epoch_080.pt
```

The evaluator reports cross-entropy loss, top-1 accuracy, and top-5 accuracy.
Pass `--save-scores results/ntu60_xsub.pkl` to store a mapping from sample name
to class-score vector.

## Validate the installation

Run the focused fidelity suite:

```console
uv run pytest
```

Or check model construction and a forward pass directly:

```console
uv run python -c "import torch; from stgcn import STGCN; m = STGCN(3, 60, {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}).eval(); print(m(torch.zeros(2, 3, 20, 25, 2)).shape)"
```

The expected printed shape is `torch.Size([2, 60])`.

### Original checkpoints

The model retains the upstream parameter names. `eval.py` accepts both a raw
upstream state dictionary and this fork's training-checkpoint dictionary, and
removes a leading `module.` prefix left by `torch.nn.DataParallel`. It loads
strictly by default so architecture or class-count mismatches are visible.

For example:

```console
uv run python eval.py \
  --config configs/ntu60_xview.yaml \
  --checkpoint /path/to/st_gcn.ntu-xview.pt
```

## Reference results

These are the **upstream repository/paper reference numbers** for the provided
ST-GCN models. They have not yet been independently reproduced by this fork.

| Model | Kinetics-Skeleton top-1 | NTU 60 cross-view | NTU 60 cross-subject |
|---|---:|---:|---:|
| ST-GCN reported upstream | **31.6%** | **88.8%** | **81.6%** |

### Independently reproduced results

No results have been recorded yet.

## Scope and repository map

```text
stgcn/model.py          ST-GCN network
stgcn/layers.py         graph and temporal convolution blocks
stgcn/graph.py          skeleton graphs and partition strategies
stgcn/data/             prepared-array dataset and preprocessing
train.py                training entry point
eval.py                 evaluation entry point
tools/prepare_*.py      skeleton-only dataset conversion
configs/                canonical benchmark configs
tests/                  fidelity and compatibility checks
```

Compared with the historical repository, this fork removes the `torchlight`
framework, processor registry, dynamic imports, two-stream extension, pose
models, OpenCV/video pipelines, tracking, demos, visualization assets, and
download scripts. None are required to reproduce the skeleton-only ST-GCN
baseline. The package depends only on PyTorch, NumPy, and PyYAML at runtime.

## Citation and provenance

This work is derived from the BSD-2-Clause-licensed
[`yysijie/st-gcn`](https://github.com/yysijie/st-gcn) repository. Copyright and
license terms are preserved in [`LICENSE`](LICENSE).

If you use ST-GCN in research, cite the original paper:

```bibtex
@inproceedings{stgcn2018aaai,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}
```

Paper: [arXiv:1801.07455](https://arxiv.org/abs/1801.07455).
