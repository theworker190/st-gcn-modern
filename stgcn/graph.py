"""Skeleton graph construction used by the original ST-GCN implementation."""

import numpy as np


class Graph:
    """Build a normalized skeleton adjacency matrix.

    ``layout`` is either ``"openpose"`` for the 18-joint Kinetics-Skeleton
    representation or ``"ntu-rgb+d"`` for the 25-joint NTU RGB+D layout.
    The partition strategies correspond to Section 3.2 of the ST-GCN paper.
    """

    def __init__(
        self,
        layout="openpose",
        strategy="uniform",
        max_hop=1,
        dilation=1,
    ):
        if max_hop < 0:
            raise ValueError("max_hop must be non-negative")
        if dilation < 1:
            raise ValueError("dilation must be at least 1")

        self.max_hop = max_hop
        self.dilation = dilation
        self._set_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        self.A = self._get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def _set_layout(self, layout):
        if layout == "openpose":
            self.num_node = 18
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
            ]
            self.center = 1
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            neighbor_1base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
                (24, 25), (25, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for i, j in neighbor_1base]
            self.center = 20
        elif layout == "ntu_edge":
            # Retained for compatibility with the upstream graph utility.
            self.num_node = 24
            neighbor_1base = [
                (1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                (8, 7), (9, 2), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (21, 22), (22, 8), (23, 24), (24, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for i, j in neighbor_1base]
            self.center = 2
        else:
            raise ValueError(f"Unsupported graph layout: {layout!r}")

        self_link = [(i, i) for i in range(self.num_node)]
        self.edge = self_link + neighbor_link

    def _get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalized = normalize_digraph(adjacency)

        if strategy == "uniform":
            return normalized[np.newaxis, ...]

        if strategy == "distance":
            partitions = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for index, hop in enumerate(valid_hop):
                partitions[index][self.hop_dis == hop] = normalized[
                    self.hop_dis == hop
                ]
            return partitions

        if strategy == "spatial":
            partitions = []
            for hop in valid_hop:
                root = np.zeros((self.num_node, self.num_node))
                close = np.zeros((self.num_node, self.num_node))
                further = np.zeros((self.num_node, self.num_node))
                for source in range(self.num_node):
                    for target in range(self.num_node):
                        if self.hop_dis[target, source] != hop:
                            continue
                        target_distance = self.hop_dis[target, self.center]
                        source_distance = self.hop_dis[source, self.center]
                        if target_distance == source_distance:
                            root[target, source] = normalized[target, source]
                        elif target_distance > source_distance:
                            close[target, source] = normalized[target, source]
                        else:
                            further[target, source] = normalized[target, source]
                if hop == 0:
                    partitions.append(root)
                else:
                    # This ordering is part of the original implementation and
                    # therefore of pretrained-checkpoint compatibility.
                    partitions.extend((root + close, further))
            return np.stack(partitions)

        raise ValueError(f"Unsupported partition strategy: {strategy!r}")


def get_hop_distance(num_node, edge, max_hop=1):
    adjacency = np.zeros((num_node, num_node))
    for source, target in edge:
        adjacency[target, source] = 1
        adjacency[source, target] = 1

    hop_distance = np.full((num_node, num_node), np.inf)
    transfer = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    reachable = np.stack(transfer) > 0
    for distance in range(max_hop, -1, -1):
        hop_distance[reachable[distance]] = distance
    return hop_distance


def normalize_digraph(adjacency):
    degree = np.sum(adjacency, axis=0)
    inverse_degree = np.zeros_like(adjacency)
    nonzero = degree > 0
    inverse_degree[nonzero, nonzero] = degree[nonzero] ** -1
    return adjacency @ inverse_degree


def normalize_undigraph(adjacency):
    degree = np.sum(adjacency, axis=0)
    inverse_sqrt_degree = np.zeros_like(adjacency)
    nonzero = degree > 0
    inverse_sqrt_degree[nonzero, nonzero] = degree[nonzero] ** -0.5
    return inverse_sqrt_degree @ adjacency @ inverse_sqrt_degree
