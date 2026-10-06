"""The original Spatial Temporal Graph Convolutional Network."""

import torch
from torch import nn
from torch.nn import functional as F

from .graph import Graph
from .layers import STGCNBlock
from .normalization import ReplicaBatchNorm1d


class STGCN(nn.Module):
    """Faithful ST-GCN baseline for skeleton-based action recognition.

    Input has shape ``(N, C, T, V, M)``: batch, channels, frames, joints,
    and people. Output has shape ``(N, num_class)``.
    """

    def __init__(
        self,
        in_channels,
        num_class,
        graph_args,
        edge_importance_weighting=True,
        dropout=0,
    ):
        super().__init__()

        self.graph = Graph(**graph_args)
        adjacency = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer("A", adjacency)

        spatial_kernel_size = adjacency.size(0)
        kernel_size = (9, spatial_kernel_size)
        self.data_bn = ReplicaBatchNorm1d(in_channels * adjacency.size(1))
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(
                    in_channels, 64, kernel_size, 1, dropout=0, residual=False
                ),
                STGCNBlock(64, 64, kernel_size, 1, dropout),
                STGCNBlock(64, 64, kernel_size, 1, dropout),
                STGCNBlock(64, 64, kernel_size, 1, dropout),
                STGCNBlock(64, 128, kernel_size, 2, dropout),
                STGCNBlock(128, 128, kernel_size, 1, dropout),
                STGCNBlock(128, 128, kernel_size, 1, dropout),
                STGCNBlock(128, 256, kernel_size, 2, dropout),
                STGCNBlock(256, 256, kernel_size, 1, dropout),
                STGCNBlock(256, 256, kernel_size, 1, dropout),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def _forward_features(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        return x, n, m

    def forward(self, x):
        x, n, m = self._forward_features(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, m, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)

    def extract_feature(self, x):
        """Return per-joint class scores and final-block features."""
        x, n, m = self._forward_features(x)
        _, channels, frames, joints = x.size()
        feature = x.view(n, m, channels, frames, joints).permute(0, 2, 3, 4, 1)
        output = self.fcn(x)
        output = output.view(n, m, -1, frames, joints).permute(0, 2, 3, 4, 1)
        return output, feature


# A small source-level alias helps readers comparing against the upstream class name.
Model = STGCN


def initialize_weights(module):
    """Apply the initialization used by the upstream training code."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)
