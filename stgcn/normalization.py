"""Replica-local BatchNorm for reproducing multi-GPU training on one device."""

import torch
from torch import nn
from torch.nn import functional as F


class _ReplicaBatchNorm:
    shards = 1

    def forward(self, x):
        if not self.training or self.shards == 1:
            return super().forward(x)
        if x.size(0) % self.shards:
            raise ValueError("BatchNorm batch dimension must be divisible by shards")
        chunks = x.chunk(self.shards, dim=0)
        # DataParallel retains running statistics only from its first replica.
        outputs = [super().forward(chunks[0])]
        outputs.extend(
            F.batch_norm(chunk, None, None, self.weight, self.bias, True, 0.0, self.eps)
            for chunk in chunks[1:]
        )
        return torch.cat(outputs, dim=0)


class ReplicaBatchNorm1d(_ReplicaBatchNorm, nn.BatchNorm1d):
    pass


class ReplicaBatchNorm2d(_ReplicaBatchNorm, nn.BatchNorm2d):
    pass


def configure_batch_norm(model, shards):
    if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1:
        raise ValueError("batch_norm_shards must be a positive integer")
    for module in model.modules():
        if isinstance(module, _ReplicaBatchNorm):
            module.shards = shards
