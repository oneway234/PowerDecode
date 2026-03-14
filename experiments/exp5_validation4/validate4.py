"""Validation 4: Power linearity assumption.

Verifies that N identical concurrent requests produce power approximately
equal to N × single-request power. Tests both prefill-heavy and decode-heavy
workloads at concurrency levels 1, 2, 3.

Each condition runs 3 rounds (averaged) to reduce GPU fluctuation noise.

Linearity coefficient in [0.85, 1.15] → PASS.

Requires: vLLM running on port 8000 (NOT the proxy).
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attribution_engine import PowerBuffer
from collectors.gpu_power import GpuPowerSamplerNVML

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("validate4")

IDLE_POWER = 21.07
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROUNDS = 3

GROUPS = {
    "P": {
        "label": "Prefill-heavy",
        "prompt": "hello " * 800,
        "max_tokens": 1,
    },
    "D": {
        "label": "Decode-heavy",
        "prompt": "hi",
        "max_tokens": 400,
    },
}


def send_request(prompt: str, max_tokens: int, result_out: dict, label: str) -> None:
    """Send a single request to vLLM."""
    client = httpx.Client(timeout=120.0)
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        response = client.post(VLLM_URL, json=request_body)
        response.raise_for_status()
        result_out["ok"] = True
    except Exception as e:
        logger.error("%s failed: %s", label, e)
        result_out["ok"] = False
    finally:
        client.close()


def measure_avg_attributable_watts(
    buffer: PowerBuffer,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    group_label: str,
) -> float:
    """Run N concurrent requests and return average attributable watts."""
    time.sleep(2.0)  # let GPU settle

    window_start = time.time()

    results = [{} for _ in range(concurrency)]
    threads = []
    for i in range(concurrency):
        label = f"{group_label}{concurrency}-t{i}"
        t = threading.Thread(
            target=send_request,
            args=(prompt, max_tokens, results[i], label),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    window_end = time.time()

    # Check all succeeded
    for r in results:
        if not r.get("ok"):
            logger.error("A request failed in %s%d", group_label, concurrency)
            return 0.0

    # Wait for trailing samples
    time.sleep(0.3)

    # Calculate avg attributable watts from power samples
    samples = buffer.query(window_start, window_end + 0.05)
    duration = window_end - window_start

    if len(samples) < 2 or duration <= 0:
        logger.warning("Not enough samples for %s%d", group_label, concurrency)
        return 0.0

    total_energy = 0.0
    for i in range(len(samples) - 1):
        t1, w1 = samples[i]
        t2, w2 = samples[i + 1]
        interval = t2 - t1
        total_energy += max(0.0, (w1 + w2) / 2 - IDLE_POWER) * interval

    avg_watts = total_energy / duration
    return avg_watts


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Start sampler + buffer ---
    sampler = GpuPowerSamplerNVML(interval_ms=10)
    sampler.start()
    logger.info("Sampler started (10ms interval)")

    time.sleep(1.0)

    buffer = PowerBuffer()

    # Feed sampler → buffer in background
    feed_stop = threading.Event()

    def feed_loop():
        last_count = 0
        while not feed_stop.is_set():
            current_count = len(sampler.samples)
            if current_count > last_count:
                for s in sampler.samples[last_count:current_count]:
                    buffer.append(s.timestamp.timestamp(), s.power_watts)
                last_count = current_count
            time.sleep(0.005)

    feeder = threading.Thread(target=feed_loop, daemon=True)
    feeder.start()

    # --- Run tests ---
    all_data = {}  # group -> concurrency -> list of avg_watts per round

    for group_key, group_cfg in GROUPS.items():
        all_data[group_key] = {}
        for n in [1, 2, 3]:
            round_watts = []
            for r in range(ROUNDS):
                logger.info(
                    "Running %s%d round %d/%d ...",
                    group_key, n, r + 1, ROUNDS,
                )
                watts = measure_avg_attributable_watts(
                    buffer,
                    group_cfg["prompt"],
                    group_cfg["max_tokens"],
                    n,
                    group_key,
                )
                round_watts.append(watts)
                logger.info(
                    "  %s%d round %d: %.2f W",
                    group_key, n, r + 1, watts,
                )
            all_data[group_key][n] = round_watts

    # Stop feeder + sampler
    feed_stop.set()
    feeder.join(timeout=2)
    sampler.stop()

    # --- Compute linearity ---
    output = {}

    for group_key, group_cfg in GROUPS.items():
        baseline = sum(all_data[group_key][1]) / ROUNDS
        group_result = {"baseline_watts": round(baseline, 2)}

        for n in [2, 3]:
            avg_watts = sum(all_data[group_key][n]) / ROUNDS
            if baseline > 0:
                linearity = avg_watts / (n * baseline)
            else:
                linearity = 0.0
            passed = 0.85 <= linearity <= 1.15
            key = f"{group_key}{n}"
            group_result[key] = {
                "avg_watts": round(avg_watts, 2),
                "linearity": round(linearity, 3),
                "passed": passed,
            }

        output[group_cfg["label"].lower().replace("-", "_")] = group_result

    # Overall pass
    all_passed = all(
        output[g][k]["passed"]
        for g in output
        for k in output[g]
        if isinstance(output[g][k], dict) and "passed" in output[g][k]
    )
    output["overall_passed"] = all_passed

    # --- Print results ---
    print()
    print("=" * 60)
    print("VALIDATION 4: Power Linearity Assumption")
    print("=" * 60)

    for group_key, group_cfg in GROUPS.items():
        group_name = group_cfg["label"]
        group_data = output[group_name.lower().replace("-", "_")]
        baseline = group_data["baseline_watts"]

        print()
        print(f"=== {group_name} Group ===")
        print(f"  {group_key}1 baseline:   {baseline:.2f} W")

        for n in [2, 3]:
            key = f"{group_key}{n}"
            d = group_data[key]
            tag = "PASS" if d["passed"] else "FAIL"
            print(f"  {key} avg watts:  {d['avg_watts']:.2f} W")
            print(f"  {key} linearity:  {d['linearity']:.3f}  {tag}")

        # Print per-round detail
        print(f"  --- Per-round detail ---")
        for n in [1, 2, 3]:
            rounds = all_data[group_key][n]
            rounds_str = ", ".join(f"{w:.2f}" for w in rounds)
            avg = sum(rounds) / len(rounds)
            print(f"  {group_key}{n}: [{rounds_str}]  avg={avg:.2f} W")

    print()
    print(f"=== Overall: {'PASS' if all_passed else 'FAIL'} ===")
    print("=" * 60)

    # --- Save result ---
    result_path = RESULTS_DIR / "result.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResult saved to {result_path}")


if __name__ == "__main__":
    main()
