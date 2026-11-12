import numpy as np

from scripts.analyze_cfs_communities import connected_components, graph_metrics


def test_connected_components_returns_disconnected_groups():
    adjacency = np.array([
        [False, True, False, False],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
    ])
    components = connected_components(adjacency)
    assert sorted(sorted(group) for group in components) == [[0, 1], [2, 3]]


def test_graph_metrics_distinguishes_directed_and_mutual_edges():
    scores = np.array([
        [np.nan, 0.8, 0.0],
        [0.7, np.nan, 0.9],
        [0.0, 0.1, np.nan],
    ])
    result = graph_metrics(scores, np.ones(3, dtype=bool), threshold=0.5)
    assert np.isclose(result["directed_edge_density"], 3 / 6)
    assert np.isclose(result["mutual_edge_density"], 2 / 6)
    assert np.isclose(result["mutual_component_ratio"], 2 / 3)
    assert np.isclose(result["mutual_largest_ratio"], 2 / 3)
    assert np.isclose(result["mutual_isolated_ratio"], 1 / 3)


def test_graph_metrics_excludes_invalid_heads():
    scores = np.array([
        [np.nan, 0.9, 0.9],
        [0.9, np.nan, 0.9],
        [0.9, 0.9, np.nan],
    ])
    result = graph_metrics(scores, np.array([True, True, False]), threshold=0.5)
    assert result["mutual_component_count"] == 1
    assert result["mutual_largest_ratio"] == 1
