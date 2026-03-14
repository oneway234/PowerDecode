"""Tests for collectors.gpu_power.GpuPowerSampler."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import pytest

from collectors.gpu_power import GpuPowerSampler, GpuPowerSamplerNVML, PowerSample


# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class TestParseLine:
    """Test the static line parser."""

    def test_normal_line(self):
        line = "  20.55 , 2026/03/12 03:06:30.009"
        sample = GpuPowerSampler._parse_line(line)
        assert isinstance(sample, PowerSample)
        assert sample.power_watts == pytest.approx(20.55)
        assert sample.timestamp == datetime(2026, 3, 12, 3, 6, 30, 9000)

    def test_integer_power(self):
        line = "120 , 2025/01/01 00:00:00.000"
        sample = GpuPowerSampler._parse_line(line)
        assert sample.power_watts == pytest.approx(120.0)

    def test_malformed_line_raises(self):
        with pytest.raises((ValueError, IndexError)):
            GpuPowerSampler._parse_line("not a valid line")


# ---------------------------------------------------------------------------
# Context-manager & start/stop tests (require nvidia-smi + GPU)
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """Check if nvidia-smi is present and a GPU is accessible."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="nvidia-smi / GPU not available",
)


@requires_gpu
class TestContextManager:
    """Test context-manager usage with real nvidia-smi."""

    def test_start_stop(self):
        sampler = GpuPowerSampler(interval_ms=100)
        sampler.start()
        assert sampler.is_running
        time.sleep(1)
        sampler.stop()
        assert not sampler.is_running
        assert len(sampler.samples) > 0

    def test_context_manager(self):
        with GpuPowerSampler(interval_ms=100) as sampler:
            assert sampler.is_running
            time.sleep(1)
        # After exiting, sampler should be stopped
        assert not sampler.is_running
        assert len(sampler.samples) > 0

    def test_double_start_raises(self):
        sampler = GpuPowerSampler()
        sampler.start()
        try:
            with pytest.raises(RuntimeError):
                sampler.start()
        finally:
            sampler.stop()


@requires_gpu
class TestDataFormat:
    """Verify collected data has correct types and structure."""

    @pytest.fixture()
    def sampler_with_data(self) -> GpuPowerSampler:
        with GpuPowerSampler(interval_ms=100) as s:
            time.sleep(1)
        return s

    def test_samples_have_correct_types(self, sampler_with_data: GpuPowerSampler):
        for sample in sampler_with_data.samples:
            assert isinstance(sample.timestamp, datetime)
            assert isinstance(sample.power_watts, float)

    def test_timestamps_are_parseable(self, sampler_with_data: GpuPowerSampler):
        for sample in sampler_with_data.samples:
            # Should be a valid datetime — converting to ISO and back must work
            iso = sample.timestamp.isoformat()
            reparsed = datetime.fromisoformat(iso)
            assert reparsed == sample.timestamp

    def test_power_values_reasonable(self, sampler_with_data: GpuPowerSampler):
        for sample in sampler_with_data.samples:
            # GPU idle ~10W, max TDP ~200W for 4060 Ti; generous range
            assert 0 < sample.power_watts < 500


@requires_gpu
class TestExport:
    """Test CSV and DataFrame export."""

    @pytest.fixture()
    def sampler_with_data(self) -> GpuPowerSampler:
        with GpuPowerSampler(interval_ms=100) as s:
            time.sleep(1)
        return s

    def test_to_csv(self, sampler_with_data: GpuPowerSampler, tmp_path: Path):
        out = sampler_with_data.to_csv(tmp_path / "power.csv")
        assert out.exists()

        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sampler_with_data.samples)
        assert set(reader.fieldnames) == {"timestamp", "power_watts"}  # type: ignore[arg-type]

        # Verify first row is parseable
        row = rows[0]
        datetime.fromisoformat(row["timestamp"])
        float(row["power_watts"])

    def test_to_dataframe(self, sampler_with_data: GpuPowerSampler):
        pd = pytest.importorskip("pandas")
        df = sampler_with_data.to_dataframe()
        assert len(df) == len(sampler_with_data.samples)
        assert list(df.columns) == ["timestamp", "power_watts"]
        assert df["power_watts"].dtype.kind == "f"  # float


@requires_gpu
class TestRealSampling:
    """Run a short real sampling session and verify end-to-end."""

    def test_two_second_sampling(self):
        with GpuPowerSampler(interval_ms=100) as sampler:
            time.sleep(2)

        # At 100ms interval for 2 seconds, expect roughly 15-25 samples
        # (some overhead for startup/teardown)
        assert len(sampler.samples) >= 10, (
            f"Expected >= 10 samples in 2s, got {len(sampler.samples)}"
        )

        # All timestamps should be within a reasonable window
        first_ts = sampler.samples[0].timestamp
        last_ts = sampler.samples[-1].timestamp
        duration = (last_ts - first_ts).total_seconds()
        assert 1.0 < duration < 4.0, f"Duration {duration}s outside expected range"


# =========================================================================
# NVML sampler tests
# =========================================================================


def _nvml_available() -> bool:
    """Check if pynvml can initialize and access a GPU."""
    try:
        from collectors.gpu_power import _nvml_acquire, _nvml_release
        import pynvml

        _nvml_acquire()
        count = pynvml.nvmlDeviceGetCount()
        _nvml_release()
        return count > 0
    except Exception:
        return False


requires_nvml = pytest.mark.skipif(
    not _nvml_available(),
    reason="pynvml / GPU not available",
)


@requires_nvml
class TestNVMLStartStop:
    """Test NVML sampler start/stop and context manager."""

    def test_start_stop(self):
        sampler = GpuPowerSamplerNVML(interval_ms=10)
        sampler.start()
        assert sampler.is_running
        time.sleep(0.5)
        sampler.stop()
        assert not sampler.is_running
        assert len(sampler.samples) > 0

    def test_context_manager(self):
        with GpuPowerSamplerNVML(interval_ms=10) as sampler:
            assert sampler.is_running
            time.sleep(0.5)
        assert not sampler.is_running
        assert len(sampler.samples) > 0

    def test_double_start_raises(self):
        sampler = GpuPowerSamplerNVML()
        sampler.start()
        try:
            with pytest.raises(RuntimeError):
                sampler.start()
        finally:
            sampler.stop()


@requires_nvml
class TestNVMLSampleCount:
    """Verify that 10ms NVML sampling produces the expected number of samples."""

    def test_two_second_sampling_10ms(self):
        with GpuPowerSamplerNVML(interval_ms=10) as sampler:
            time.sleep(2)

        # 2s at 10ms interval -> ~200 samples, allow ±20%
        n = len(sampler.samples)
        assert 160 <= n <= 240, (
            f"Expected ~200 samples (±20%) in 2s at 10ms, got {n}"
        )


@requires_nvml
class TestNVMLDataFormat:
    """Verify NVML-collected data has correct types and structure."""

    @pytest.fixture()
    def sampler_with_data(self) -> GpuPowerSamplerNVML:
        with GpuPowerSamplerNVML(interval_ms=10) as s:
            time.sleep(0.5)
        return s

    def test_samples_are_power_sample(self, sampler_with_data: GpuPowerSamplerNVML):
        for sample in sampler_with_data.samples:
            assert isinstance(sample, PowerSample)

    def test_samples_have_correct_types(self, sampler_with_data: GpuPowerSamplerNVML):
        for sample in sampler_with_data.samples:
            assert isinstance(sample.timestamp, datetime)
            assert isinstance(sample.power_watts, float)

    def test_power_values_reasonable(self, sampler_with_data: GpuPowerSamplerNVML):
        for sample in sampler_with_data.samples:
            assert 0 < sample.power_watts < 500


@requires_nvml
class TestNVMLExport:
    """Test CSV and DataFrame export for the NVML sampler."""

    @pytest.fixture()
    def sampler_with_data(self) -> GpuPowerSamplerNVML:
        with GpuPowerSamplerNVML(interval_ms=10) as s:
            time.sleep(0.5)
        return s

    def test_to_csv(self, sampler_with_data: GpuPowerSamplerNVML, tmp_path: Path):
        out = sampler_with_data.to_csv(tmp_path / "power_nvml.csv")
        assert out.exists()

        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sampler_with_data.samples)
        assert set(reader.fieldnames) == {"timestamp", "power_watts"}  # type: ignore[arg-type]

        row = rows[0]
        datetime.fromisoformat(row["timestamp"])
        float(row["power_watts"])

    def test_to_dataframe(self, sampler_with_data: GpuPowerSamplerNVML):
        pd = pytest.importorskip("pandas")
        df = sampler_with_data.to_dataframe()
        assert len(df) == len(sampler_with_data.samples)
        assert list(df.columns) == ["timestamp", "power_watts"]
        assert df["power_watts"].dtype.kind == "f"
