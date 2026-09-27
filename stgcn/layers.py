"""Graph-convolution and spatial-temporal building blocks for ST-GCN."""

import torch
from torch import nn


class ConvTemporalGraphical(nn.Module):
    """Apply the spatial graph convolution from the original ST-GCN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, adjacency):
        if adjacency.size(0) != self.kernel_size:
            raise ValueError(
                f"Expected {self.kernel_size} adjacency partitions, "
                f"received {adjacency.size(0)}"
            )
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, adjacency)
        return x.contiguous(), adjacency


class STGCNBlock(nn.Module):
    """One spatial graph-convolution followed by a temporal convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()
        if len(kernel_size) != 2 or kernel_size[0] % 2 != 1:
            raise ValueError("kernel_size must contain an odd temporal size")

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda _x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda value: value
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacency):
        residual = self.residual(x)
        x, adjacency = self.gcn(x, adjacency)
        x = self.tcn(x) + residual
        return self.relu(x), adjacency
