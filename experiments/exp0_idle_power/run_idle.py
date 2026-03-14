"""Measure GPU idle power consumption over 5 minutes."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collectors.gpu_power import GpuPowerSampler

DURATION_SECONDS = 300
INTERVAL_MS = 100
PROGRESS_EVERY = 30  # seconds


def _disable_screensaver() -> None:
    """Disable screensaver and screen blanking during experiment."""
    try:
        subprocess.run(["xset", "s", "off"], check=False, capture_output=True)
        subprocess.run(["xset", "-dpms"], check=False, capture_output=True)
        subprocess.run(["xdg-screensaver", "reset"], check=False, capture_output=True)
        print("  Screensaver disabled.")
    except FileNotFoundError:
        print("  Warning: xset/xdg-screensaver not found, skipping.")


def _enable_screensaver() -> None:
    """Re-enable screensaver and screen blanking after experiment."""
    try:
        subprocess.run(["xset", "s", "on"], check=False, capture_output=True)
        subprocess.run(["xset", "+dpms"], check=False, capture_output=True)
        print("  Screensaver re-enabled.")
    except FileNotFoundError:
        pass


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _disable_screensaver()
    print(f"Starting GPU idle power measurement for {DURATION_SECONDS} seconds ...")

    sampler = GpuPowerSampler(interval_ms=INTERVAL_MS)
    sampler.start()

    start_time = time.monotonic()
    next_report = PROGRESS_EVERY

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= DURATION_SECONDS:
                break
            if elapsed >= next_report:
                print(f"  Progress: {int(next_report)} seconds / {DURATION_SECONDS} seconds sampled")
                next_report += PROGRESS_EVERY
            # Sleep in short bursts so we don't overshoot the progress ticks
            time.sleep(min(1.0, DURATION_SECONDS - elapsed))
    finally:
        sampler.stop()

    print(f"Sampling complete. Collected {len(sampler.samples)} samples.")

    # --- Write CSV ---
    csv_path = results_dir / "idle_power.csv"
    sampler.to_csv(csv_path)
    print(f"CSV written to {csv_path}")

    # --- Compute statistics and write JSON ---
    watts = [s.power_watts for s in sampler.samples]

    if not watts:
        print("ERROR: no samples collected – is nvidia-smi available?")
        sys.exit(1)

    summary = {
        "mean_watts": round(statistics.mean(watts), 4),
        "min_watts": round(min(watts), 4),
        "max_watts": round(max(watts), 4),
        "std_watts": round(statistics.stdev(watts), 4) if len(watts) > 1 else 0.0,
        "sample_count": len(watts),
        "duration_seconds": DURATION_SECONDS,
    }

    json_path = results_dir / "idle_power.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"JSON written to {json_path}")
    print()
    print(json.dumps(summary, indent=2))

    _enable_screensaver()


if __name__ == "__main__":
    main()
