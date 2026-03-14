#!/usr/bin/env python3
"""
T-004: Align GPU power samples with request timing data.

Reads GPU power CSV (from GpuPowerSampler.to_csv()) and request log JSON
(from run_requests.py), slices power samples into each request's time window,
computes per-request power statistics, and outputs aligned results plus a
per-test_name summary.

Outputs:
  - experiments/exp1_prefill_decode/results/aligned_data.json
  - experiments/exp1_prefill_decode/results/summary.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_power_csv(filepath: str | Path) -> list[dict]:
    """Load GPU power CSV into a sorted list of {timestamp: datetime, power_watts: float}.

    The CSV produced by GpuPowerSampler.to_csv() uses ISO-format timestamps.
    nvidia-smi raw format ("2026/03/12 02:34:36.459") is also accepted.
    """
    samples: list[dict] = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["timestamp"].strip()
            ts = _parse_timestamp(ts_str)
            samples.append({
                "timestamp": ts,
                "power_watts": float(row["power_watts"]),
            })
    # Ensure chronological order
    samples.sort(key=lambda s: s["timestamp"])
    return samples


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse a timestamp string in either ISO format or nvidia-smi format.

    Accepted formats:
      - ISO 8601: "2026-03-12T02:34:36.459000" (from to_csv / isoformat)
      - nvidia-smi: "2026/03/12 02:34:36.459"
    """
    # Try ISO format first (most common from to_csv)
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass
    # Try nvidia-smi format
    try:
        return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        pass
    # Try without fractional seconds
    return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")


def load_request_log(filepath: str | Path) -> list[dict]:
    """Load request log JSON.

    start_time and end_time are Unix epoch floats — convert to datetime for
    comparison with power timestamps.
    """
    with open(filepath) as f:
        records = json.load(f)

    for rec in records:
        rec["start_dt"] = datetime.fromtimestamp(rec["start_time"])
        rec["end_dt"] = datetime.fromtimestamp(rec["end_time"])
    return records


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------


def _extract_timestamps(samples: list[dict]) -> list[datetime]:
    """Return a list of timestamps (same order as samples) for bisect lookups."""
    return [s["timestamp"] for s in samples]


def slice_power_for_request(
    samples: list[dict],
    timestamps: list[datetime],
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """Return power samples whose timestamp falls within [start_dt, end_dt].

    If no samples fall in the window (request too short), return the single
    nearest sample.
    """
    lo = bisect.bisect_left(timestamps, start_dt)
    hi = bisect.bisect_right(timestamps, end_dt)

    sliced = samples[lo:hi]

    if sliced:
        return sliced

    # Fallback: find the nearest sample to the midpoint of the request window
    if not samples:
        return []

    midpoint = start_dt + (end_dt - start_dt) / 2
    # Find insertion point for midpoint
    idx = bisect.bisect_left(timestamps, midpoint)
    # Choose the closer of idx-1 and idx
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(samples):
        candidates.append(idx)

    if not candidates:
        return []

    nearest_idx = min(candidates, key=lambda i: abs(timestamps[i] - midpoint))
    return [samples[nearest_idx]]


def compute_power_stats(
    power_samples: list[dict],
    duration_sec: float,
) -> dict:
    """Compute avg_watts, peak_watts, duration_sec, energy_joules."""
    if not power_samples:
        return {
            "avg_watts": None,
            "peak_watts": None,
            "duration_sec": duration_sec,
            "energy_joules": None,
            "power_sample_count": 0,
        }

    watts = [s["power_watts"] for s in power_samples]
    avg_w = mean(watts)
    peak_w = max(watts)
    energy_j = avg_w * duration_sec

    return {
        "avg_watts": round(avg_w, 3),
        "peak_watts": round(peak_w, 3),
        "duration_sec": round(duration_sec, 6),
        "energy_joules": round(energy_j, 3),
        "power_sample_count": len(power_samples),
    }


def align_data(
    power_csv_path: str | Path,
    request_log_path: str | Path,
) -> tuple[list[dict], dict]:
    """Main alignment routine.

    Returns:
        (aligned_records, summary_by_test_name)
    """
    power_samples = load_power_csv(power_csv_path)
    requests = load_request_log(request_log_path)

    if not power_samples:
        print("WARNING: No power samples found in CSV.")
    if not requests:
        print("WARNING: No requests found in log.")

    timestamps = _extract_timestamps(power_samples)

    aligned: list[dict] = []

    for req in requests:
        sliced = slice_power_for_request(
            power_samples, timestamps, req["start_dt"], req["end_dt"],
        )
        stats = compute_power_stats(sliced, req["latency"])

        record = {
            "test_name": req["test_name"],
            "round": req["round"],
            "start_time": req["start_time"],
            "end_time": req["end_time"],
            "latency": req["latency"],
            "prompt_tokens": req["prompt_tokens"],
            "completion_tokens": req["completion_tokens"],
            **stats,
        }
        aligned.append(record)

    # Build summary grouped by test_name
    summary = _build_summary(aligned)

    return aligned, summary


def _safe_stdev(values: list[float]) -> float | None:
    """Return stdev if there are at least 2 values, else None."""
    if len(values) < 2:
        return None
    return round(stdev(values), 3)


def _build_summary(aligned: list[dict]) -> dict:
    """Group by test_name and compute mean/stdev for key metrics."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in aligned:
        groups[rec["test_name"]].append(rec)

    summary: dict[str, dict] = {}

    for test_name, records in sorted(groups.items()):
        avg_watts_vals = [r["avg_watts"] for r in records if r["avg_watts"] is not None]
        peak_watts_vals = [r["peak_watts"] for r in records if r["peak_watts"] is not None]
        energy_vals = [r["energy_joules"] for r in records if r["energy_joules"] is not None]
        latency_vals = [r["latency"] for r in records]
        duration_vals = [r["duration_sec"] for r in records]
        sample_counts = [r["power_sample_count"] for r in records]

        summary[test_name] = {
            "num_rounds": len(records),
            "avg_watts_mean": round(mean(avg_watts_vals), 3) if avg_watts_vals else None,
            "avg_watts_std": _safe_stdev(avg_watts_vals),
            "peak_watts_mean": round(mean(peak_watts_vals), 3) if peak_watts_vals else None,
            "peak_watts_std": _safe_stdev(peak_watts_vals),
            "energy_joules_mean": round(mean(energy_vals), 3) if energy_vals else None,
            "energy_joules_std": _safe_stdev(energy_vals),
            "latency_mean": round(mean(latency_vals), 3) if latency_vals else None,
            "latency_std": _safe_stdev(latency_vals),
            "duration_sec_mean": round(mean(duration_vals), 3) if duration_vals else None,
            "duration_sec_std": _safe_stdev(duration_vals),
            "avg_power_sample_count": round(mean(sample_counts), 1) if sample_counts else None,
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align GPU power data with request timing logs.",
    )
    parser.add_argument(
        "--power-csv",
        required=True,
        help="Path to GPU power CSV (from GpuPowerSampler.to_csv())",
    )
    parser.add_argument(
        "--request-log",
        default=str(RESULTS_DIR / "request_log.json"),
        help="Path to request log JSON (default: results/request_log.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory for output files (default: results/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = output_dir / "aligned_data.json"
    summary_path = output_dir / "summary.json"

    print(f"Power CSV:    {args.power_csv}")
    print(f"Request log:  {args.request_log}")
    print(f"Output dir:   {output_dir}")
    print()

    aligned, summary = align_data(args.power_csv, args.request_log)

    # Write aligned data
    aligned_path.write_text(json.dumps(aligned, indent=2))
    print(f"Aligned data ({len(aligned)} records) -> {aligned_path}")

    # Write summary
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary ({len(summary)} test groups) -> {summary_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print("Summary by test_name")
    print(f"{'='*60}")
    for test_name, stats in summary.items():
        print(f"\n  {test_name} ({stats['num_rounds']} rounds):")
        print(f"    avg_watts:    {stats['avg_watts_mean']} +/- {stats['avg_watts_std']} W")
        print(f"    peak_watts:   {stats['peak_watts_mean']} +/- {stats['peak_watts_std']} W")
        print(f"    energy:       {stats['energy_joules_mean']} +/- {stats['energy_joules_std']} J")
        print(f"    latency:      {stats['latency_mean']} +/- {stats['latency_std']} s")
        print(f"    power samples/request: {stats['avg_power_sample_count']}")


if __name__ == "__main__":
    main()
