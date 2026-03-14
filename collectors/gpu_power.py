"""GPU power sampler using nvidia-smi.

Launches nvidia-smi in continuous sampling mode (default 100ms interval)
and collects timestamped power readings. Supports both context-manager
and explicit start/stop usage.
"""

from __future__ import annotations

import csv
import io
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Self


@dataclass
class PowerSample:
    """A single GPU power reading."""

    timestamp: datetime
    power_watts: float


class GpuPowerSampler:
    """Collect GPU power draw via nvidia-smi.

    Usage as context manager::

        with GpuPowerSampler() as sampler:
            # ... workload ...
        df = sampler.to_dataframe()

    Usage with explicit start/stop::

        sampler = GpuPowerSampler()
        sampler.start()
        # ... workload ...
        sampler.stop()
        sampler.to_csv("power.csv")
    """

    def __init__(
        self,
        interval_ms: int = 100,
        gpu_index: int = 0,
    ) -> None:
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.samples: list[PowerSample] = []

        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.stop()

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch nvidia-smi sampling in a background subprocess."""
        if self._process is not None:
            raise RuntimeError("Sampler is already running")

        self._stop_event.clear()
        self.samples.clear()

        cmd = [
            "nvidia-smi",
            f"--query-gpu=power.draw,timestamp",
            f"--format=csv,noheader,nounits",
            f"-lms",
            str(self.interval_ms),
            f"-i",
            str(self.gpu_index),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the reader thread to finish."""
        if self._process is None:
            return

        self._stop_event.set()
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)

        self._process = None
        self._reader_thread = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def to_csv(self, filepath: str | Path) -> Path:
        """Write collected samples to a CSV file.

        Returns the resolved Path for convenience.
        """
        filepath = Path(filepath)
        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "power_watts"])
            for s in self.samples:
                writer.writerow([s.timestamp.isoformat(), s.power_watts])
        return filepath

    def to_dataframe(self):  # noqa: ANN201 – pandas is optional
        """Return samples as a pandas DataFrame."""
        import pandas as pd

        rows = [
            {"timestamp": s.timestamp, "power_watts": s.power_watts}
            for s in self.samples
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Continuously read stdout from nvidia-smi and parse samples."""
        assert self._process is not None
        assert self._process.stdout is not None

        for line in self._process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = self._parse_line(line)
                self.samples.append(sample)
            except (ValueError, IndexError):
                # Skip malformed lines (e.g. header remnants)
                continue

    @staticmethod
    def _parse_line(line: str) -> PowerSample:
        """Parse a single nvidia-smi CSV output line.

        Expected format: ``"  20.55 , 2026/03/12 03:06:30.009"``
        """
        parts = line.split(",", maxsplit=1)
        power_str = parts[0].strip()
        ts_str = parts[1].strip()

        power_watts = float(power_str)
        timestamp = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")

        return PowerSample(timestamp=timestamp, power_watts=power_watts)


# ======================================================================
# NVML lifecycle — reference-counted init/shutdown
# ======================================================================

_nvml_refcount = 0
_nvml_lock = threading.Lock()


def _nvml_acquire() -> None:
    """Increment NVML ref count; call nvmlInit on first acquire."""
    global _nvml_refcount
    import pynvml

    with _nvml_lock:
        if _nvml_refcount == 0:
            pynvml.nvmlInit()
        _nvml_refcount += 1


def _nvml_release() -> None:
    """Decrement NVML ref count; call nvmlShutdown when it hits zero."""
    global _nvml_refcount
    import pynvml

    with _nvml_lock:
        _nvml_refcount -= 1
        if _nvml_refcount <= 0:
            _nvml_refcount = 0
            pynvml.nvmlShutdown()


# ======================================================================
# NVML-based high-frequency sampler
# ======================================================================


class GpuPowerSamplerNVML:
    """Collect GPU power draw via pynvml at high frequency.

    Uses NVML directly (no subprocess) to achieve sampling intervals as
    low as 5–10 ms, which is essential for capturing short-lived power
    spikes (e.g. during LLM prefill).

    Usage as context manager::

        with GpuPowerSamplerNVML(interval_ms=10) as sampler:
            # ... workload ...
        df = sampler.to_dataframe()

    Usage with explicit start/stop::

        sampler = GpuPowerSamplerNVML(interval_ms=10)
        sampler.start()
        # ... workload ...
        sampler.stop()
        sampler.to_csv("power.csv")
    """

    def __init__(
        self,
        interval_ms: int = 10,
        gpu_index: int = 0,
    ) -> None:
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.samples: list[PowerSample] = []

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.stop()

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin high-frequency NVML power sampling in a background thread."""
        if self._thread is not None:
            raise RuntimeError("Sampler is already running")

        import pynvml

        _nvml_acquire()
        self._nvml_initialized = True

        self._stop_event.clear()
        self.samples.clear()

        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and release NVML."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

        if self._nvml_initialized:
            _nvml_release()
            self._nvml_initialized = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def to_csv(self, filepath: str | Path) -> Path:
        """Write collected samples to a CSV file.

        Returns the resolved Path for convenience.
        """
        filepath = Path(filepath)
        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "power_watts"])
            for s in self.samples:
                writer.writerow([s.timestamp.isoformat(), s.power_watts])
        return filepath

    def to_dataframe(self):  # noqa: ANN201 – pandas is optional
        """Return samples as a pandas DataFrame."""
        import pandas as pd

        rows = [
            {"timestamp": s.timestamp, "power_watts": s.power_watts}
            for s in self.samples
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """High-frequency sampling loop using pynvml."""
        import pynvml
        import time as _time

        interval_s = self.interval_ms / 1000.0

        while not self._stop_event.is_set():
            t0 = _time.perf_counter()
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_watts = milliwatts / 1000.0
                timestamp = datetime.now()
                self.samples.append(
                    PowerSample(timestamp=timestamp, power_watts=power_watts)
                )
            except pynvml.NVMLError:
                # Skip transient NVML errors rather than crashing the thread
                pass

            # Sleep for the remainder of the interval, accounting for
            # time spent in the NVML call itself.
            elapsed = _time.perf_counter() - t0
            sleep_time = interval_s - elapsed
            if sleep_time > 0:
                _time.sleep(sleep_time)


# ======================================================================
# Multi-GPU sampler (tensor parallel / multi-card)
# ======================================================================


class MultiGpuPowerSamplerNVML:
    """Sample multiple GPUs simultaneously, summing power into a single stream.

    Each sample's ``power_watts`` is the sum across all GPUs.  This is the
    correct aggregation for vLLM tensor-parallel mode where one request
    spans all cards.

    API is identical to GpuPowerSamplerNVML (samples, start, stop,
    is_running, to_csv, to_dataframe, context manager).

    Usage::

        sampler = MultiGpuPowerSamplerNVML(gpu_indices=[0, 1, 2, 3])
        sampler.start()
        # ... workload ...
        sampler.stop()
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        interval_ms: int = 10,
    ) -> None:
        self.interval_ms = interval_ms
        self.samples: list[PowerSample] = []

        # Auto-detect GPU count if not specified
        if gpu_indices is None:
            try:
                import pynvml
                _nvml_acquire()
                count = pynvml.nvmlDeviceGetCount()
                _nvml_release()
                gpu_indices = list(range(count))
            except Exception:
                gpu_indices = [0]

        self.gpu_indices = gpu_indices
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> None:
        """Begin sampling all GPUs in a single background thread."""
        if self._thread is not None:
            raise RuntimeError("Sampler is already running")

        import pynvml

        _nvml_acquire()
        self._nvml_initialized = True

        self._stop_event.clear()
        self.samples.clear()

        self._handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices
        ]

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and release NVML."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

        if self._nvml_initialized:
            _nvml_release()
            self._nvml_initialized = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def to_csv(self, filepath: str | Path) -> Path:
        filepath = Path(filepath)
        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "power_watts"])
            for s in self.samples:
                writer.writerow([s.timestamp.isoformat(), s.power_watts])
        return filepath

    def to_dataframe(self):
        import pandas as pd
        rows = [
            {"timestamp": s.timestamp, "power_watts": s.power_watts}
            for s in self.samples
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _sample_loop(self) -> None:
        """Read all GPU handles each tick, sum watts into one PowerSample."""
        import pynvml
        import time as _time

        interval_s = self.interval_ms / 1000.0

        while not self._stop_event.is_set():
            t0 = _time.perf_counter()
            try:
                total_watts = 0.0
                for handle in self._handles:
                    milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
                    total_watts += milliwatts / 1000.0
                self.samples.append(
                    PowerSample(timestamp=datetime.now(), power_watts=total_watts)
                )
            except pynvml.NVMLError:
                pass

            elapsed = _time.perf_counter() - t0
            sleep_time = interval_s - elapsed
            if sleep_time > 0:
                _time.sleep(sleep_time)
