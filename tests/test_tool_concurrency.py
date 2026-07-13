"""Concurrent tool dispatch: independence classification, overlap, and safety.

These lock in issue #65's contract for the ``ToolExecutor``: a contiguous run of
provably read-only calls in one assistant message overlaps (so a fan-out turn's
wall-clock approaches its slowest call, not the sum), while every dependent or
side-effecting call stays strictly sequential and every result still maps back to
its own ``tool_call_id`` regardless of finish order.
"""

import json
import threading
import time

import pytest

from my_coding_agent.engine.tool_execution.concurrency import (
    is_parallel_safe,
    is_read_only_command,
    max_tool_concurrency,
)
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry

# ── independence classification ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "cat file.txt",
        "grep -n needle src.py",
        "head -n 50 log.txt",
        "cat a.txt | grep foo | head -5",
        "wc -l data.csv",
        "ls -la /tmp",
        "jq . config.json",
        "rg pattern .",
    ],
)
def test_read_only_command_accepts_safe_reads(command):
    assert is_read_only_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "cat a.txt > b.txt",  # redirection writes
        "cat a.txt >> b.txt",  # append writes
        "grep x f | tee out.txt",  # tee is not allow-listed
        "cat a.txt; rm b.txt",  # command chaining
        "cat a.txt && rm b.txt",  # && chaining (contains &)
        "echo $(rm x)",  # command substitution
        "cat `rm x`",  # backtick substitution
        "sort -o out.txt in.txt",  # sort's output-file flag
        "sed -i s/a/b/ f",  # in-place edit
        "awk '{print}' f",  # awk can write via redirection
        "curl http://example.com",  # network reach
        "wget http://example.com",  # writes a file by default
        "find . -delete",  # deletes
        "rm file.txt",  # not read-only at all
        "",  # empty
    ],
)
def test_read_only_command_rejects_unsafe(command):
    assert is_read_only_command(command) is False


def test_is_parallel_safe_by_tool():
    assert is_parallel_safe("read_file", {"file_path": "/x"}) is True
    assert is_parallel_safe("fetch_web", {"url": "http://x"}) is True
    assert is_parallel_safe("bash", {"command": "cat f"}) is True
    assert is_parallel_safe("bash", {"command": "rm f"}) is False
    assert is_parallel_safe("write_file", {"file_path": "/x", "content": "y"}) is False
    # read_tool_artifact makes its own LLM call; delegate can mutate anything.
    assert is_parallel_safe("read_tool_artifact", {"tool_call_id": "c"}) is False
    assert is_parallel_safe("delegate", {"task": "do"}) is False
    assert is_parallel_safe("unknown_tool", {}) is False


def test_max_tool_concurrency_env(monkeypatch):
    monkeypatch.delenv("MCA_TOOL_MAX_CONCURRENCY", raising=False)
    assert max_tool_concurrency() >= 1
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "7")
    assert max_tool_concurrency() == 7
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "0")
    assert max_tool_concurrency() == 1  # clamped: 0 would disable the pool
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "junk")
    assert max_tool_concurrency() >= 1  # non-integer falls back to default


# ── concurrent dispatch harness ───────────────────────────────────────────────


class _Probe:
    """Tracks live concurrency across patched tool bodies, thread-safely."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: list[str] = []
        self.max_active = 0
        self.write_overlapped = False

    def enter(self, name: str) -> None:
        with self._lock:
            self._active.append(name)
            self.max_active = max(self.max_active, len(self._active))
            if len(self._active) > 1 and "write_file" in self._active:
                self.write_overlapped = True

    def leave(self, name: str) -> None:
        with self._lock:
            self._active.remove(name)


def _tool_call(name, args, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _patch_sleeping_tool(monkeypatch, probe, method, sleep_s=0.15):
    """Patch a registry method to register concurrency and sleep, then return a str."""

    def body(self, **kwargs):
        probe.enter(method)
        try:
            time.sleep(sleep_s)
            return f"{method}:{sorted(kwargs.items())}"
        finally:
            probe.leave(method)

    monkeypatch.setattr(ToolsRegistry, method, body)


def test_independent_reads_overlap(bare_executor, monkeypatch):
    """Three independent read_file calls run concurrently, not back-to-back."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "4")
    probe = _Probe()
    _patch_sleeping_tool(monkeypatch, probe, "read_file", sleep_s=0.2)
    bare_executor.tool_calls = [
        _tool_call("read_file", {"file_path": f"/f{i}"}, f"c{i}") for i in range(3)
    ]
    start = time.monotonic()
    messages, records = bare_executor.run()
    elapsed = time.monotonic() - start

    assert probe.max_active >= 2  # genuinely overlapped
    assert elapsed < 0.55  # ~max(0.2) + overhead, well under the 0.6 s serial sum
    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2"]
    assert all(m["status"] == "success" for m in messages)
    assert len(records) == 3


def test_results_map_to_ids_under_out_of_order_finish(bare_executor, monkeypatch):
    """Even when later calls finish first, each result maps to its own id in order."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "4")
    probe = _Probe()

    def body(self, file_path):
        # Earlier calls sleep longer, so completion order is the reverse of
        # submission order — the assembled results must still be in call order.
        idx = int(file_path.rsplit("f", 1)[1])
        probe.enter("read_file")
        try:
            time.sleep(0.05 * (3 - idx))
            return f"content-of-f{idx}"
        finally:
            probe.leave("read_file")

    monkeypatch.setattr(ToolsRegistry, "read_file", body)
    bare_executor.tool_calls = [
        _tool_call("read_file", {"file_path": f"/f{i}"}, f"c{i}") for i in range(3)
    ]
    messages, _ = bare_executor.run()

    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2"]
    for i, m in enumerate(messages):
        assert json.loads(m["content"])["output"] == f"content-of-f{i}"


def test_write_calls_stay_serial(bare_executor, monkeypatch):
    """Side-effecting write_file calls never overlap — each is its own barrier."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "4")
    probe = _Probe()
    _patch_sleeping_tool(monkeypatch, probe, "write_file", sleep_s=0.05)
    bare_executor.tool_calls = [
        _tool_call("write_file", {"file_path": f"/f{i}", "content": "x"}, f"c{i}")
        for i in range(3)
    ]
    messages, _ = bare_executor.run()

    assert probe.max_active == 1  # strictly one at a time
    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2"]


def test_barrier_between_reads_preserves_order_and_isolation(
    bare_executor, monkeypatch
):
    """A write between reads is a barrier: reads overlap, the write runs alone."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "4")
    probe = _Probe()
    _patch_sleeping_tool(monkeypatch, probe, "read_file", sleep_s=0.1)
    _patch_sleeping_tool(monkeypatch, probe, "write_file", sleep_s=0.1)
    bare_executor.tool_calls = [
        _tool_call("read_file", {"file_path": "/a"}, "c0"),
        _tool_call("read_file", {"file_path": "/b"}, "c1"),
        _tool_call("write_file", {"file_path": "/c", "content": "x"}, "c2"),
        _tool_call("read_file", {"file_path": "/d"}, "c3"),
    ]
    messages, _ = bare_executor.run()

    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2", "c3"]
    assert probe.max_active >= 2  # the two leading reads overlapped
    assert probe.write_overlapped is False  # the write never overlapped anything


def test_concurrency_of_one_disables_overlap(bare_executor, monkeypatch):
    """MCA_TOOL_MAX_CONCURRENCY=1 keeps the read group correct but serial."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "1")
    probe = _Probe()
    _patch_sleeping_tool(monkeypatch, probe, "read_file", sleep_s=0.02)
    bare_executor.tool_calls = [
        _tool_call("read_file", {"file_path": f"/f{i}"}, f"c{i}") for i in range(3)
    ]
    messages, _ = bare_executor.run()

    assert probe.max_active == 1  # no overlap
    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2"]


# ── recorder timing under concurrency ─────────────────────────────────────────


class _RecordingRecorder:
    """Captures before/after_tool invocations for assertions."""

    def __init__(self):
        self.before_calls = 0
        self.after_timings: list[tuple | None] = []

    def before_tool(self, name, args):
        self.before_calls += 1
        return args

    def after_tool(self, name, args, result, ok, error, timing=None):
        self.after_timings.append(timing)
        return result


def test_parallel_path_passes_true_per_call_timing(bare_executor, monkeypatch):
    """Overlapped calls record explicit timing and bypass the shared pending slot."""
    monkeypatch.setenv("MCA_TOOL_MAX_CONCURRENCY", "4")
    recorder = _RecordingRecorder()
    bare_executor.llm._recorder = recorder
    _patch_sleeping_tool(monkeypatch, _Probe(), "read_file", sleep_s=0.05)
    bare_executor.tool_calls = [
        _tool_call("read_file", {"file_path": f"/f{i}"}, f"c{i}") for i in range(2)
    ]
    bare_executor.run()

    # before_tool (the single-slot stamp) is skipped on the concurrent path;
    # each after_tool instead carries a (start, end, started_at) bracket.
    assert recorder.before_calls == 0
    assert len(recorder.after_timings) == 2
    for timing in recorder.after_timings:
        assert timing is not None
        start_mono, end_mono, started_at = timing
        assert end_mono > start_mono  # real elapsed duration
        assert isinstance(started_at, str)
        assert started_at


def test_single_call_uses_sequential_recorder_path(bare_executor, monkeypatch):
    """A lone read (no read-only neighbour) keeps the before/after pending path."""
    recorder = _RecordingRecorder()
    bare_executor.llm._recorder = recorder
    _patch_sleeping_tool(monkeypatch, _Probe(), "read_file", sleep_s=0.01)
    bare_executor.tool_calls = [_tool_call("read_file", {"file_path": "/only"}, "c0")]
    bare_executor.run()

    assert recorder.before_calls == 1
    assert recorder.after_timings == [None]  # sequential path passes no explicit timing
