"""Machine-wide resource sampler — background daemon thread with a ring buffer.

``ResourceSampler`` samples RAM/CPU/GPU/network/disk on a fixed interval into a
bounded in-memory ring buffer, and summarizes the samples inside a
``[start, end]`` monotonic window on demand. It is passive capture
infrastructure (records reality, never controls execution — CONTRIBUTE.md's
observability/passive-vs-active split) and requires no elevated privileges:
RAM/CPU/network/disk come from ``psutil``; GPU utilization on Apple Silicon
comes from parsing ``ioreg -r -d 1 -c IOAccelerator`` (``powermetrics`` needs
root, so this is the only sudo-free GPU signal on macOS).

``get_sampler()`` returns one process-wide singleton, shared by the main agent
and any in-process subagents (machine-wide data is identical for all of
them); ``start``/``stop`` are reference-counted so nested agent lifecycles
don't tear the thread down under each other.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import Any

import psutil

from .schema import (
    RESOURCES_CPU_PCT,
    RESOURCES_DISK_BYTES,
    RESOURCES_GAUGE_AVG,
    RESOURCES_GAUGE_PEAK,
    RESOURCES_GPU_PCT,
    RESOURCES_MACHINE_WIDE,
    RESOURCES_NET_BYTES,
    RESOURCES_RAM_PCT,
    RESOURCES_SAMPLE_COUNT,
)

logger = logging.getLogger(__name__)

_SAMPLE_INTERVAL_S = 1.0
# Longer than any plausible single event window (an LLM call rarely exceeds a
# few minutes), so a window lookup almost never misses samples it should see.
_RING_SECONDS = 300.0
_GPU_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
# Absolute path (not "ioreg" on $PATH) — avoids a partial-executable-path
# resolution and keeps the subprocess call fully static/untainted by input.
_IOREG_PATH = "/usr/sbin/ioreg"


class _Sample:
    """One instantaneous reading. Plain slots class — created ~once/second."""

    __slots__ = ("cpu_pct", "disk_bytes", "gpu_pct", "net_bytes", "ram_pct", "t")

    def __init__(
        self,
        t: float,
        ram_pct: float,
        cpu_pct: float,
        gpu_pct: float | None,
        net_bytes: int,
        disk_bytes: int,
    ) -> None:
        self.t = t
        self.ram_pct = ram_pct
        self.cpu_pct = cpu_pct
        self.gpu_pct = gpu_pct
        self.net_bytes = net_bytes
        self.disk_bytes = disk_bytes


def _summarize_samples(samples: list[_Sample]) -> dict[str, Any]:
    """Gauges (RAM/CPU/GPU) as avg+peak; counters (net/disk) as edge deltas."""
    n = len(samples)
    ram_vals = [s.ram_pct for s in samples]
    cpu_vals = [s.cpu_pct for s in samples]
    gpu_vals = [s.gpu_pct for s in samples if s.gpu_pct is not None]
    result: dict[str, Any] = {
        RESOURCES_MACHINE_WIDE: True,
        RESOURCES_SAMPLE_COUNT: n,
        RESOURCES_RAM_PCT: {
            RESOURCES_GAUGE_AVG: round(sum(ram_vals) / n, 1),
            RESOURCES_GAUGE_PEAK: round(max(ram_vals), 1),
        },
        RESOURCES_CPU_PCT: {
            RESOURCES_GAUGE_AVG: round(sum(cpu_vals) / n, 1),
            RESOURCES_GAUGE_PEAK: round(max(cpu_vals), 1),
        },
        RESOURCES_NET_BYTES: samples[-1].net_bytes - samples[0].net_bytes,
        RESOURCES_DISK_BYTES: samples[-1].disk_bytes - samples[0].disk_bytes,
    }
    if gpu_vals:
        result[RESOURCES_GPU_PCT] = {
            RESOURCES_GAUGE_AVG: round(sum(gpu_vals) / len(gpu_vals), 1),
            RESOURCES_GAUGE_PEAK: round(max(gpu_vals), 1),
        }
    return result


class ResourceSampler:
    """Background daemon thread sampling machine-wide resource use.

    Any exception during a sampling tick or an on-demand snapshot disables
    further capture permanently and logs one warning — the run continues,
    subsequent calls to :meth:`summarize_window` / :meth:`session_rollup`
    simply return ``None`` (spec: "Resource capture degrades gracefully").
    """

    def __init__(
        self,
        interval_s: float = _SAMPLE_INTERVAL_S,
        ring_seconds: float = _RING_SECONDS,
    ) -> None:
        self._interval = interval_s
        self._ring_capacity = max(int(ring_seconds / interval_s), 1)
        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._refcount = 0
        self._disabled = False
        self._gpu_available = True
        # Running rollup folded from samples evicted out of the ring, so a
        # long run's session-wide rollup stays correct after eviction.
        self._rollup_count = 0
        self._rollup_ram_sum = 0.0
        self._rollup_ram_peak = 0.0
        self._rollup_cpu_sum = 0.0
        self._rollup_cpu_peak = 0.0
        self._rollup_gpu_sum = 0.0
        self._rollup_gpu_peak = 0.0
        self._rollup_gpu_count = 0
        self._first_net_bytes: int | None = None
        self._last_net_bytes: int | None = None
        self._first_disk_bytes: int | None = None
        self._last_disk_bytes: int | None = None

    def start(self) -> None:
        """Start the sampler thread if not already running (ref-counted)."""
        with self._lock:
            self._refcount += 1
            if self._disabled or self._thread is not None:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="resource-sampler", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Release one reference; stop the thread once the last one drops."""
        thread: threading.Thread | None = None
        with self._lock:
            self._refcount = max(self._refcount - 1, 0)
            if self._refcount == 0 and self._thread is not None:
                self._stop_event.set()
                thread = self._thread
                self._thread = None
        if thread is not None:
            thread.join(timeout=self._interval * 2)

    def _run(self) -> None:
        # First cpu_percent() call is meaningless (no baseline yet); prime it
        # before the loop so the first real tick reflects the interval.
        psutil.cpu_percent(interval=None)
        while not self._stop_event.wait(self._interval):
            self._tick()

    def _tick(self) -> None:
        try:
            sample = self._take_sample()
        except Exception:
            logger.warning(
                "resource sampler failed; disabling further capture", exc_info=True
            )
            with self._lock:
                self._disabled = True
            return
        with self._lock:
            self._add_sample_locked(sample)

    def _take_sample(self) -> _Sample:
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        gpu_pct = self._read_gpu_util() if self._gpu_available else None
        return _Sample(
            t=time.monotonic(),
            ram_pct=vm.percent,
            cpu_pct=cpu_pct,
            gpu_pct=gpu_pct,
            net_bytes=(net.bytes_sent + net.bytes_recv) if net else 0,
            disk_bytes=(disk.read_bytes + disk.write_bytes) if disk else 0,
        )

    def _read_gpu_util(self) -> float | None:
        """Parse ``Device Utilization %`` from ``ioreg``; ``None`` on any failure.

        The ``ioreg`` output format is undocumented and macOS-only; any parse
        or subprocess failure disables only the GPU field (``_gpu_available``),
        never the other resources.
        """
        try:
            out = subprocess.run(  # noqa: S603 -- fixed absolute path, fixed argv, no shell
                [_IOREG_PATH, "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            ).stdout
        except Exception:
            self._gpu_available = False
            return None
        match = _GPU_UTIL_RE.search(out)
        if match is None:
            self._gpu_available = False
            return None
        return float(match.group(1))

    def _add_sample_locked(self, sample: _Sample) -> None:
        if self._first_net_bytes is None:
            self._first_net_bytes = sample.net_bytes
            self._first_disk_bytes = sample.disk_bytes
        self._last_net_bytes = sample.net_bytes
        self._last_disk_bytes = sample.disk_bytes
        self._samples.append(sample)
        while len(self._samples) > self._ring_capacity:
            self._fold_into_rollup_locked(self._samples.pop(0))

    def _fold_into_rollup_locked(self, sample: _Sample) -> None:
        self._rollup_count += 1
        self._rollup_ram_sum += sample.ram_pct
        self._rollup_ram_peak = max(self._rollup_ram_peak, sample.ram_pct)
        self._rollup_cpu_sum += sample.cpu_pct
        self._rollup_cpu_peak = max(self._rollup_cpu_peak, sample.cpu_pct)
        if sample.gpu_pct is not None:
            self._rollup_gpu_sum += sample.gpu_pct
            self._rollup_gpu_peak = max(self._rollup_gpu_peak, sample.gpu_pct)
            self._rollup_gpu_count += 1

    def summarize_window(self, start: float, end: float) -> dict[str, Any] | None:
        """Summarize samples in monotonic ``[start, end]``, or ``None`` if unavailable.

        Falls back to a single on-demand snapshot when the window is shorter
        than one sampling interval (no ring samples fall inside it), so fast
        tool calls still get figures.
        """
        with self._lock:
            if self._disabled:
                return None
            in_window = [s for s in self._samples if start <= s.t <= end]
        if not in_window:
            try:
                in_window = [self._take_sample()]
            except Exception:
                logger.warning(
                    "resource sampler snapshot failed; disabling further capture",
                    exc_info=True,
                )
                with self._lock:
                    self._disabled = True
                return None
        return _summarize_samples(in_window)

    def session_rollup(self) -> dict[str, Any] | None:
        """Summarize the whole run so far: ring samples plus the folded rollup."""
        with self._lock:
            if self._disabled:
                return None
            samples = list(self._samples)
            count = self._rollup_count
            ram_sum, ram_peak = self._rollup_ram_sum, self._rollup_ram_peak
            cpu_sum, cpu_peak = self._rollup_cpu_sum, self._rollup_cpu_peak
            gpu_sum, gpu_peak = self._rollup_gpu_sum, self._rollup_gpu_peak
            gpu_count = self._rollup_gpu_count
            first_net, last_net = self._first_net_bytes, self._last_net_bytes
            first_disk, last_disk = self._first_disk_bytes, self._last_disk_bytes
        for s in samples:
            count += 1
            ram_sum += s.ram_pct
            ram_peak = max(ram_peak, s.ram_pct)
            cpu_sum += s.cpu_pct
            cpu_peak = max(cpu_peak, s.cpu_pct)
            if s.gpu_pct is not None:
                gpu_sum += s.gpu_pct
                gpu_peak = max(gpu_peak, s.gpu_pct)
                gpu_count += 1
        if count == 0:
            return None
        result: dict[str, Any] = {
            RESOURCES_MACHINE_WIDE: True,
            RESOURCES_SAMPLE_COUNT: count,
            RESOURCES_RAM_PCT: {
                RESOURCES_GAUGE_AVG: round(ram_sum / count, 1),
                RESOURCES_GAUGE_PEAK: round(ram_peak, 1),
            },
            RESOURCES_CPU_PCT: {
                RESOURCES_GAUGE_AVG: round(cpu_sum / count, 1),
                RESOURCES_GAUGE_PEAK: round(cpu_peak, 1),
            },
            RESOURCES_NET_BYTES: (
                (last_net - first_net)
                if first_net is not None and last_net is not None
                else 0
            ),
            RESOURCES_DISK_BYTES: (
                (last_disk - first_disk)
                if first_disk is not None and last_disk is not None
                else 0
            ),
        }
        if gpu_count:
            result[RESOURCES_GPU_PCT] = {
                RESOURCES_GAUGE_AVG: round(gpu_sum / gpu_count, 1),
                RESOURCES_GAUGE_PEAK: round(gpu_peak, 1),
            }
        return result


_singleton_lock = threading.Lock()
_singleton: ResourceSampler | None = None


def get_sampler() -> ResourceSampler:
    """Return the one process-wide sampler, shared by main agent and subagents."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = ResourceSampler()
        return _singleton
