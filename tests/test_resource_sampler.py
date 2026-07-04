"""Tests for the node-resource-monitoring background sampler.

Covers window summarization math (avg/peak/deltas), the sub-interval
on-demand snapshot fallback, GPU parse-failure tolerance, and fault
containment (a sampler exception disables further capture without raising).
"""

from __future__ import annotations

from my_coding_agent.observability.sampler import ResourceSampler, _Sample


def _sample(t, ram=10.0, cpu=20.0, gpu=30.0, net=1000, disk=2000):
    return _Sample(
        t=t, ram_pct=ram, cpu_pct=cpu, gpu_pct=gpu, net_bytes=net, disk_bytes=disk
    )


class TestSummarizeWindow:
    def test_avg_and_peak_over_multiple_samples(self):
        s = ResourceSampler(interval_s=1.0)
        s._samples = [
            _sample(1.0, ram=10.0, cpu=20.0, gpu=30.0, net=1000, disk=2000),
            _sample(2.0, ram=30.0, cpu=60.0, gpu=90.0, net=1500, disk=2500),
        ]
        result = s.summarize_window(1.0, 2.0)
        assert result["sample_count"] == 2
        assert result["ram_pct"] == {"avg": 20.0, "peak": 30.0}
        assert result["cpu_pct"] == {"avg": 40.0, "peak": 60.0}
        assert result["gpu_pct"] == {"avg": 60.0, "peak": 90.0}
        assert result["net_bytes"] == 500
        assert result["disk_bytes"] == 500
        assert result["machine_wide"] is True

    def test_gpu_absent_when_no_sample_has_gpu(self):
        s = ResourceSampler(interval_s=1.0)
        s._samples = [_sample(1.0, gpu=None), _sample(2.0, gpu=None)]
        result = s.summarize_window(1.0, 2.0)
        assert "gpu_pct" not in result

    def test_window_outside_ring_falls_back_to_snapshot(self, monkeypatch):
        s = ResourceSampler(interval_s=1.0)
        s._samples = []
        monkeypatch.setattr(
            s, "_take_sample", lambda: _sample(5.0, ram=42.0, cpu=11.0, gpu=None)
        )
        result = s.summarize_window(100.0, 100.5)
        assert result["sample_count"] == 1
        assert result["ram_pct"]["avg"] == 42.0
        # Single-sample deltas are zero — acceptable for fast, cheap calls.
        assert result["net_bytes"] == 0
        assert result["disk_bytes"] == 0

    def test_disabled_sampler_returns_none(self):
        s = ResourceSampler(interval_s=1.0)
        s._disabled = True
        assert s.summarize_window(0.0, 1.0) is None


class TestFaultContainment:
    def test_snapshot_failure_disables_capture_and_returns_none(self, monkeypatch):
        s = ResourceSampler(interval_s=1.0)
        s._samples = []

        def _boom():
            raise RuntimeError("ioreg exploded")

        monkeypatch.setattr(s, "_take_sample", _boom)
        result = s.summarize_window(0.0, 1.0)
        assert result is None
        assert s._disabled is True
        # A second call, even without the monkeypatched failure, stays disabled.
        assert s.summarize_window(0.0, 1.0) is None

    def test_tick_failure_disables_capture(self, monkeypatch):
        s = ResourceSampler(interval_s=1.0)

        def _boom():
            raise RuntimeError("psutil exploded")

        monkeypatch.setattr(s, "_take_sample", _boom)
        s._tick()
        assert s._disabled is True
        assert s.session_rollup() is None


class TestGpuParseFailure:
    def test_unparseable_ioreg_output_yields_absent_gpu_field(self, monkeypatch):
        s = ResourceSampler(interval_s=1.0)

        class _Proc:
            stdout = "no gpu info here"

        monkeypatch.setattr(
            "my_coding_agent.observability.sampler.subprocess.run",
            lambda *a, **k: _Proc(),
        )
        assert s._read_gpu_util() is None
        assert s._gpu_available is False

    def test_ioreg_subprocess_failure_yields_absent_gpu_field(self, monkeypatch):
        s = ResourceSampler(interval_s=1.0)

        def _raise(*a, **k):
            raise OSError("no ioreg on this host")

        monkeypatch.setattr(
            "my_coding_agent.observability.sampler.subprocess.run", _raise
        )
        assert s._read_gpu_util() is None
        assert s._gpu_available is False


class TestSessionRollup:
    def test_folds_ring_and_evicted_samples(self):
        s = ResourceSampler(interval_s=1.0, ring_seconds=2.0)  # capacity 2
        for i in range(5):
            sample = _sample(float(i), ram=float(i * 10), cpu=5.0, gpu=None)
            s._add_sample_locked(sample)
        rollup = s.session_rollup()
        assert rollup["sample_count"] == 5
        assert rollup["ram_pct"]["peak"] == 40.0
        assert rollup["net_bytes"] == 1000 - 1000  # constant net in fixture -> 0
        assert "gpu_pct" not in rollup

    def test_no_samples_returns_none(self):
        s = ResourceSampler(interval_s=1.0)
        assert s.session_rollup() is None


class TestStartStopRefcounting:
    def test_stop_without_start_does_not_raise(self):
        s = ResourceSampler(interval_s=1.0)
        s.stop()  # refcount floor at 0, no thread to join

    def test_nested_start_stop_keeps_thread_alive_until_last_stop(self):
        s = ResourceSampler(interval_s=100.0)  # long interval, tick never fires
        s.start()
        s.start()
        assert s._thread is not None
        s.stop()
        assert s._thread is not None  # still one outstanding ref
        s.stop()
        assert s._thread is None
