"""Tests for the resource rollup section of print_run_summary.

(node-resource-monitoring)
"""

from __future__ import annotations

import re

from my_coding_agent.utils.terminal_ui import print_run_summary

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(capsys, **kwargs):
    defaults = dict(
        steps=1,
        max_steps=5,
        stop_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    defaults.update(kwargs)
    print_run_summary(**defaults)
    out = capsys.readouterr().err
    return _ANSI_RE.sub("", out)


def test_no_resource_section_when_rollup_absent(capsys):
    plain = _capture(capsys, resource_rollup=None)
    assert "RAM (machine)" not in plain
    assert "CPU (machine)" not in plain


def test_resource_section_renders_ram_cpu_net_disk(capsys):
    rollup = {
        "ram_pct": {"avg": 40.5, "peak": 60.2},
        "cpu_pct": {"avg": 20.1, "peak": 55.9},
        "net_bytes": 2 * 1024 * 1024,
        "disk_bytes": 1024 * 1024,
    }
    plain = _capture(capsys, resource_rollup=rollup)
    assert "RAM (machine)" in plain
    assert "avg 40.5%" in plain
    assert "peak 60.2%" in plain
    assert "CPU (machine)" in plain
    assert "NET / DISK (machine)" in plain
    assert "2.0 MB" in plain
    assert "1.0 MB" in plain
    assert "GPU (machine)" not in plain


def test_resource_section_includes_gpu_when_present(capsys):
    rollup = {
        "ram_pct": {"avg": 1.0, "peak": 2.0},
        "cpu_pct": {"avg": 3.0, "peak": 4.0},
        "gpu_pct": {"avg": 5.0, "peak": 6.0},
        "net_bytes": 0,
        "disk_bytes": 0,
    }
    plain = _capture(capsys, resource_rollup=rollup)
    assert "GPU (machine)" in plain
    assert "avg 5.0%" in plain
