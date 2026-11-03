from typing import cast

import torch
from torch_geometric.data import HeteroData

from edurec.datasets import ElearningDataModule
from scripts import common


def _graph() -> HeteroData:
    graph = HeteroData()
    graph["item"].num_nodes = 3
    graph["attr::concept"].num_nodes = 4
    graph["item", "concept", "attr::concept"].edge_index = torch.tensor(
        [[0, 1, 1], [0, 2, 3]]
    )
    graph["attr::concept", "rev_concept", "item"].edge_index = torch.tensor(
        [[0, 2, 3], [0, 1, 1]]
    )
    graph["item", "ref::prerequisite", "item"].edge_index = torch.tensor([[0], [2]])
    graph["item", "rev_ref::prerequisite", "item"].edge_index = torch.tensor(
        [[2], [0]]
    )
    return graph


class _Adapter:
    def __init__(self, graph: HeteroData):
        self.knowledge_graph = graph


class _DataModule:
    def __init__(self, graph: HeteroData):
        self.adapter = _Adapter(graph)


def _module(graph: HeteroData) -> ElearningDataModule:
    """Stand-in for a prepared data module with an arbitrary graph."""
    return cast(ElearningDataModule, _DataModule(graph))


def test_graph_statistics_separates_forward_and_reverse_relations():
    stats = common.graph_statistics(_graph())

    assert stats["nodes"] == {"item": 3, "attr::concept": 4}
    assert stats["num_nodes"] == 7
    assert stats["forward_edges"] == {
        "item -concept-> attr::concept": 3,
        "item -ref::prerequisite-> item": 1,
    }
    assert stats["reverse_edges"] == {
        "attr::concept -rev_concept-> item": 3,
        "item -rev_ref::prerequisite-> item": 1,
    }
    assert stats["num_forward_edges"] == 4
    assert stats["num_reverse_edges"] == 4
    assert stats["num_relations"] == 4


def test_print_graph_summary_reports_nodes_and_edges(capsys, monkeypatch):
    monkeypatch.setattr(common, "EdurecDataAdapter", _Adapter)

    common.print_graph_summary("TEST", _module(_graph()))
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == (
        "[TEST] Graph: nodes=7, edges=8 (forward=4, reverse=4), relations=4"
    )
    assert lines[1] == "[TEST] Graph nodes: item=3, attr::concept=4"
    assert "item -concept-> attr::concept=3" in lines[2]
    assert "item -ref::prerequisite-> item=1" in lines[2]


def test_print_graph_summary_handles_a_graph_without_relations(capsys, monkeypatch):
    monkeypatch.setattr(common, "EdurecDataAdapter", _Adapter)
    graph = HeteroData()
    graph["item"].num_nodes = 5

    common.print_graph_summary("TEST", _module(graph))
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == (
        "[TEST] Graph: nodes=5, edges=0 (forward=0, reverse=0), relations=0"
    )
    assert len(lines) == 2


def test_print_graph_summary_skips_other_adapters(capsys):
    common.print_graph_summary("TEST", _module(_graph()))

    assert capsys.readouterr().out == ""
