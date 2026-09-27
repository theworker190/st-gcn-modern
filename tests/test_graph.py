import numpy as np
import pytest

from stgcn.graph import Graph, get_hop_distance, normalize_digraph


@pytest.mark.parametrize(
    ("layout", "joint_count"),
    [("openpose", 18), ("ntu-rgb+d", 25)],
)
def test_spatial_graph_shape_and_normalization(layout, joint_count):
    graph = Graph(layout=layout, strategy="spatial")

    assert graph.A.shape == (3, joint_count, joint_count)
    np.testing.assert_allclose(graph.A.sum(axis=0).sum(axis=0), np.ones(joint_count))
    assert np.all(np.diag(graph.A[0]) > 0)


def test_partition_strategies_preserve_normalized_adjacency():
    uniform = Graph(layout="ntu-rgb+d", strategy="uniform").A
    distance = Graph(layout="ntu-rgb+d", strategy="distance").A
    spatial = Graph(layout="ntu-rgb+d", strategy="spatial").A

    np.testing.assert_allclose(distance.sum(axis=0), uniform[0])
    np.testing.assert_allclose(spatial.sum(axis=0), uniform[0])


def test_hop_distance_and_directed_normalization():
    edges = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    distance = get_hop_distance(3, edges, max_hop=2)
    np.testing.assert_array_equal(
        distance,
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
    )

    adjacency = np.array([[1.0, 1.0], [0.0, 1.0]])
    normalized = normalize_digraph(adjacency)
    np.testing.assert_allclose(normalized.sum(axis=0), np.ones(2))
