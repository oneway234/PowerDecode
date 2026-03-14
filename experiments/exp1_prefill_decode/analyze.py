#!/usr/bin/env python3
"""
T-005: Analyze & Visualize — Determine weighted method viability.

Reads aligned_data.json and summary.json (produced by align_data.py),
computes per-token power metrics, generates comparison charts, and
outputs a viability conclusion.

Outputs (all under experiments/exp1_prefill_decode/results/):
  - fig1_avg_power_bar.png      — Bar chart of avg watts per test group
  - fig2_power_timeseries.png   — Per-round power time series (if data allows)
  - fig3_watts_per_token.png    — watts/token comparison (prefill vs decode)
  - conclusion.json             — Machine-readable conclusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"
THRESHOLD_PCT = 15.0

TEST_A = "A_pure_prefill"
TEST_B = "B_pure_decode"
TEST_C = "C_mixed_baseline"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(filepath: Path) -> dict | list:
    with open(filepath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze(aligned: list[dict], summary: dict) -> dict:
    """Compute all analysis metrics and return the conclusion dict."""

    # -- Extract summary stats for A and B ---------------------------------
    sum_a = summary.get(TEST_A)
    sum_b = summary.get(TEST_B)

    if sum_a is None or sum_b is None:
        sys.exit(
            f"ERROR: summary.json must contain both '{TEST_A}' and '{TEST_B}'. "
            f"Found keys: {list(summary.keys())}"
        )

    avg_watts_a = sum_a["avg_watts_mean"]
    avg_watts_b = sum_b["avg_watts_mean"]

    if avg_watts_a is None or avg_watts_b is None:
        sys.exit("ERROR: avg_watts_mean is None for test A or B — no power data?")

    # -- Difference percentage (relative to the lower of the two) ----------
    base = min(avg_watts_a, avg_watts_b)
    diff_pct = abs(avg_watts_a - avg_watts_b) / base * 100.0 if base > 0 else 0.0

    # -- watts/token -------------------------------------------------------
    recs_a = [r for r in aligned if r["test_name"] == TEST_A]
    recs_b = [r for r in aligned if r["test_name"] == TEST_B]

    # For prefill: each request processes prompt_tokens during prefill
    wpt_prefill_vals = [
        r["avg_watts"] / r["prompt_tokens"]
        for r in recs_a
        if r["avg_watts"] is not None and r["prompt_tokens"] and r["prompt_tokens"] > 0
    ]
    # For decode: each request generates completion_tokens during decode
    wpt_decode_vals = [
        r["avg_watts"] / r["completion_tokens"]
        for r in recs_b
        if r["avg_watts"] is not None and r["completion_tokens"] and r["completion_tokens"] > 0
    ]

    w_prefill = mean(wpt_prefill_vals) if wpt_prefill_vals else None
    w_decode = mean(wpt_decode_vals) if wpt_decode_vals else None

    ratio = w_prefill / w_decode if (w_prefill and w_decode and w_decode != 0) else None

    viable = diff_pct > THRESHOLD_PCT

    if viable:
        recommendation = (
            f"Weighted method is VIABLE. A vs B power difference is {diff_pct:.1f}% "
            f"(>{THRESHOLD_PCT}%). Use W_prefill={w_prefill:.4f} W/tok and "
            f"W_decode={w_decode:.4f} W/tok (ratio={ratio:.3f}) for energy attribution."
        )
    else:
        recommendation = (
            f"Weighted method is NOT viable. A vs B power difference is only {diff_pct:.1f}% "
            f"(<={THRESHOLD_PCT}%). Fall back to pure time-slice energy attribution."
        )

    conclusion = {
        "test_a_avg_watts": round(avg_watts_a, 3),
        "test_b_avg_watts": round(avg_watts_b, 3),
        "difference_pct": round(diff_pct, 2),
        "threshold_pct": THRESHOLD_PCT,
        "weighted_method_viable": viable,
        "w_prefill": round(w_prefill, 6) if w_prefill is not None else None,
        "w_decode": round(w_decode, 6) if w_decode is not None else None,
        "w_ratio": round(ratio, 4) if ratio is not None else None,
        "recommendation": recommendation,
    }

    return conclusion


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _safe_stdev(values: list[float]) -> float:
    """Return stdev if >= 2 values, else 0."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def plot_avg_power_bar(summary: dict, output_path: Path) -> None:
    """Fig 1: Bar chart of average watts per test group with error bars."""
    test_names = []
    means = []
    stds = []

    for name in [TEST_A, TEST_B, TEST_C]:
        if name not in summary:
            continue
        s = summary[name]
        if s["avg_watts_mean"] is None:
            continue
        test_names.append(name)
        means.append(s["avg_watts_mean"])
        stds.append(s["avg_watts_std"] if s["avg_watts_std"] is not None else 0.0)

    if not test_names:
        print("  [Fig 1] Skipped — no data.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(test_names, means, yerr=stds, capsize=6, color=colors[: len(test_names)],
                  edgecolor="black", linewidth=0.7, alpha=0.85)

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                f"{m:.1f} W", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Power (Watts)")
    ax.set_title("Fig 1: Average GPU Power by Test Group")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Fig 1] Saved -> {output_path}")


def plot_power_timeseries(aligned: list[dict], output_path: Path) -> None:
    """Fig 2: Per-round power time series for each test group."""
    # Only plot if we have multiple power samples per request on average
    has_multi_sample = any(r.get("power_sample_count", 0) > 1 for r in aligned)

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {"A_pure_prefill": "o", "B_pure_decode": "s", "C_mixed_baseline": "^"}
    colors = {"A_pure_prefill": "#4C72B0", "B_pure_decode": "#DD8452", "C_mixed_baseline": "#55A868"}

    for name in [TEST_A, TEST_B, TEST_C]:
        recs = [r for r in aligned if r["test_name"] == name and r["avg_watts"] is not None]
        if not recs:
            continue
        rounds = [r["round"] for r in recs]
        watts = [r["avg_watts"] for r in recs]
        ax.plot(rounds, watts, marker=markers.get(name, "o"), label=name,
                color=colors.get(name, "gray"), linewidth=1.5, markersize=6)

    ax.set_xlabel("Round")
    ax.set_ylabel("Average Power (Watts)")
    subtitle = "(per-request avg watts)" if has_multi_sample else "(single sample per request)"
    ax.set_title(f"Fig 2: Power per Round by Test Group {subtitle}")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Fig 2] Saved -> {output_path}")


def plot_watts_per_token(aligned: list[dict], output_path: Path) -> None:
    """Fig 3: watts/token comparison — prefill vs decode."""
    recs_a = [r for r in aligned if r["test_name"] == TEST_A]
    recs_b = [r for r in aligned if r["test_name"] == TEST_B]

    wpt_prefill = [
        r["avg_watts"] / r["prompt_tokens"]
        for r in recs_a
        if r["avg_watts"] is not None and r["prompt_tokens"] and r["prompt_tokens"] > 0
    ]
    wpt_decode = [
        r["avg_watts"] / r["completion_tokens"]
        for r in recs_b
        if r["avg_watts"] is not None and r["completion_tokens"] and r["completion_tokens"] > 0
    ]

    if not wpt_prefill and not wpt_decode:
        print("  [Fig 3] Skipped — no per-token data.")
        return

    labels = []
    means = []
    stds = []
    colors = []

    if wpt_prefill:
        labels.append("W/prefill_token\n(Test A)")
        means.append(mean(wpt_prefill))
        stds.append(_safe_stdev(wpt_prefill))
        colors.append("#4C72B0")

    if wpt_decode:
        labels.append("W/decode_token\n(Test B)")
        means.append(mean(wpt_decode))
        stds.append(_safe_stdev(wpt_decode))
        colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.7, alpha=0.85)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.001,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Watts per Token")
    ax.set_title("Fig 3: Power per Token — Prefill vs Decode")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Fig 3] Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T-005: Analyze experiment 1 data and determine weighted method viability.",
    )
    parser.add_argument(
        "--aligned-data",
        default=str(RESULTS_DIR / "aligned_data.json"),
        help="Path to aligned_data.json (default: results/aligned_data.json)",
    )
    parser.add_argument(
        "--summary",
        default=str(RESULTS_DIR / "summary.json"),
        help="Path to summary.json (default: results/summary.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory for output files (default: results/)",
    )
    args = parser.parse_args()

    aligned_path = Path(args.aligned_data)
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aligned data: {aligned_path}")
    print(f"Summary:      {summary_path}")
    print(f"Output dir:   {output_dir}")
    print()

    # -- Load data ---------------------------------------------------------
    aligned = load_json(aligned_path)
    summary = load_json(summary_path)

    # -- Core analysis -----------------------------------------------------
    print("=" * 60)
    print("Running analysis ...")
    print("=" * 60)

    conclusion = analyze(aligned, summary)

    # -- Print conclusion to stdout ----------------------------------------
    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Test A ({TEST_A}) avg power : {conclusion['test_a_avg_watts']} W")
    print(f"  Test B ({TEST_B}) avg power : {conclusion['test_b_avg_watts']} W")
    print(f"  Difference                  : {conclusion['difference_pct']:.2f}%")
    print(f"  Threshold                   : {conclusion['threshold_pct']}%")
    print(f"  Weighted method viable      : {conclusion['weighted_method_viable']}")
    print(f"  W_prefill (W/token)         : {conclusion['w_prefill']}")
    print(f"  W_decode  (W/token)         : {conclusion['w_decode']}")
    print(f"  W_prefill / W_decode ratio  : {conclusion['w_ratio']}")
    print()
    print(f"  >> {conclusion['recommendation']}")
    print()

    # -- Write conclusion.json ---------------------------------------------
    conclusion_path = output_dir / "conclusion.json"
    conclusion_path.write_text(json.dumps(conclusion, indent=2))
    print(f"Conclusion written -> {conclusion_path}")

    # -- Plots -------------------------------------------------------------
    print()
    print("Generating plots ...")
    plot_avg_power_bar(summary, output_dir / "fig1_avg_power_bar.png")
    plot_power_timeseries(aligned, output_dir / "fig2_power_timeseries.png")
    plot_watts_per_token(aligned, output_dir / "fig3_watts_per_token.png")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
