#!/usr/bin/env python3
"""
Master orchestration script for Experiment 1: Prefill vs Decode.

Runs the full pipeline in one shot:
  1. Start GPU power sampling (background thread via GpuPowerSampler)
  2. Execute all three request test groups (foreground)
  3. Stop power sampling and save CSV
  4. Align power data with request timing
  5. Analyze results and generate plots
  6. Print conclusion.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from collectors.gpu_power import GpuPowerSamplerNVML as GpuPowerSampler
from experiments.exp1_prefill_decode.run_requests import main as run_requests_main, RESULTS_DIR
from experiments.exp1_prefill_decode.align_data import align_data
from experiments.exp1_prefill_decode.analyze import (
    analyze,
    load_json,
    plot_avg_power_bar,
    plot_power_timeseries,
    plot_watts_per_token,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POWER_CSV = RESULTS_DIR / "gpu_power.csv"
REQUEST_LOG = RESULTS_DIR / "request_log.json"
ALIGNED_DATA = RESULTS_DIR / "aligned_data.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
CONCLUSION_JSON = RESULTS_DIR / "conclusion.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _disable_screensaver() -> None:
    """Disable screensaver and screen blanking during experiment."""
    try:
        subprocess.run(["xset", "s", "off"], check=False, capture_output=True)
        subprocess.run(["xset", "-dpms"], check=False, capture_output=True)
        subprocess.run(["xdg-screensaver", "reset"], check=False, capture_output=True)
        print("Screensaver disabled.")
    except FileNotFoundError:
        print("Warning: xset/xdg-screensaver not found, skipping.")


def _enable_screensaver() -> None:
    """Re-enable screensaver and screen blanking after experiment."""
    try:
        subprocess.run(["xset", "s", "on"], check=False, capture_output=True)
        subprocess.run(["xset", "+dpms"], check=False, capture_output=True)
        print("Screensaver re-enabled.")
    except FileNotFoundError:
        pass


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _disable_screensaver()

    # ------------------------------------------------------------------
    # Step 1 & 2: Power sampling (background) + requests (foreground)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/4: Starting GPU power sampler + running requests")
    print("=" * 60)

    with GpuPowerSampler() as sampler:
        # run_requests.main() creates RESULTS_DIR and writes request_log.json
        run_requests_main()

    # Sampler is now stopped (exited context manager)
    print(f"\nCollected {len(sampler.samples)} power samples.")

    # ------------------------------------------------------------------
    # Step 2b: Save power CSV
    # ------------------------------------------------------------------
    sampler.to_csv(POWER_CSV)
    print(f"Power data saved -> {POWER_CSV}")

    # ------------------------------------------------------------------
    # Step 3: Align data
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2/4: Aligning power data with request timing")
    print("=" * 60)

    aligned, summary = align_data(POWER_CSV, REQUEST_LOG)

    ALIGNED_DATA.write_text(json.dumps(aligned, indent=2))
    print(f"Aligned data ({len(aligned)} records) -> {ALIGNED_DATA}")

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Summary ({len(summary)} test groups) -> {SUMMARY_JSON}")

    # ------------------------------------------------------------------
    # Step 4: Analyze
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/4: Analyzing results")
    print("=" * 60)

    conclusion = analyze(aligned, summary)

    CONCLUSION_JSON.write_text(json.dumps(conclusion, indent=2))
    print(f"Conclusion written -> {CONCLUSION_JSON}")

    # ------------------------------------------------------------------
    # Step 5: Generate plots
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4/4: Generating plots")
    print("=" * 60)

    plot_avg_power_bar(summary, RESULTS_DIR / "fig1_avg_power_bar.png")
    plot_power_timeseries(aligned, RESULTS_DIR / "fig2_power_timeseries.png")
    plot_watts_per_token(aligned, RESULTS_DIR / "fig3_watts_per_token.png")

    # ------------------------------------------------------------------
    # Final: Print conclusion
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(json.dumps(conclusion, indent=2))
    print()
    print("Experiment 1 complete.")

    _enable_screensaver()


if __name__ == "__main__":
    main()
