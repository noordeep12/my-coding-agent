"""Tests for pipeline/registry.py — node enumeration and graph validation."""

from __future__ import annotations

from my_coding_agent.pipeline.registry import (
    CANONICAL_ORDER,
    list_node_types,
    validate_runnable,
)


def test_list_node_types_lists_exactly_the_registered_node_types():
    names = [t["name"] for t in list_node_types()]
    assert names == list(CANONICAL_ORDER)
    assert names == [
        "context_guard",
        "llm_call",
        "tool_dispatch",
        "anomaly_detect",
        "finalize_step",
    ]


def _canonical_graph():
    nodes = [{"id": f"n{i}", "type": t} for i, t in enumerate(CANONICAL_ORDER)]
    edges = [
        {"from": nodes[i]["id"], "to": nodes[i + 1]["id"]}
        for i in range(len(nodes) - 1)
    ]
    return nodes, edges, nodes[0]["id"], nodes[-1]["id"]


def test_validate_runnable_accepts_the_canonical_chain():
    nodes, edges, start, end = _canonical_graph()
    assert validate_runnable(nodes, edges, start, end) is None


def test_validate_runnable_rejects_missing_start_end():
    nodes, edges, _start, _end = _canonical_graph()
    assert validate_runnable(nodes, edges, None, None) is not None


def test_validate_runnable_rejects_unregistered_node_type():
    nodes, edges, start, end = _canonical_graph()
    nodes[0]["type"] = "made_up_node"
    assert validate_runnable(nodes, edges, start, end) is not None


def test_validate_runnable_rejects_wrong_order():
    nodes = [{"id": "a", "type": "llm_call"}, {"id": "b", "type": "context_guard"}]
    edges = [{"from": "a", "to": "b"}]
    assert validate_runnable(nodes, edges, "a", "b") is not None


def test_validate_runnable_rejects_orphan_node():
    nodes, edges, start, end = _canonical_graph()
    nodes.append({"id": "orphan", "type": "llm_call"})
    assert validate_runnable(nodes, edges, start, end) is not None


def test_validate_runnable_rejects_dangling_edge():
    nodes, edges, start, end = _canonical_graph()
    edges[0] = {"from": edges[0]["from"], "to": "does-not-exist"}
    assert validate_runnable(nodes, edges, start, end) is not None


def test_validate_runnable_rejects_branching():
    nodes, edges, start, end = _canonical_graph()
    edges.append({"from": start, "to": end})
    assert validate_runnable(nodes, edges, start, end) is not None
