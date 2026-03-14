"""PowerDecode Multi-Model Benchmark.

Usage:
    1. Start vLLM with a model on port 8000
    2. Start proxy.py on port 8001
    3. python3 benchmark.py

Auto-detects model from vLLM. Sends standardized prompts through proxy,
waits for attribution, then prints energy/cost comparison table.

Results append to benchmark_results.json — run once per model, then
use --report to print the comparison across all models.
"""

import json
import sys
import time
from pathlib import Path

import httpx

PROXY_URL = "http://localhost:8001/v1/chat/completions"
VLLM_URL = "http://localhost:8000/v1/models"
STATS_URL = "http://localhost:8001/stats/recent"
RESULT_PATH = Path(__file__).resolve().parent / "benchmark_results.json"

# Standardized test cases: (label, prompt, max_tokens)
TEST_CASES = [
    ("short-in/short-out", "Say hello.", 20),
    ("short-in/long-out", "Write a 200-word essay about energy efficiency.", 300),
    ("long-in/short-out", "Summarize: " + "The quick brown fox jumps. " * 200, 20),
    ("long-in/long-out", "Expand on: " + "AI inference costs matter. " * 100, 300),
]

WARMUP_ROUNDS = 3
ROUNDS_PER_CASE = 3


def _detect_model() -> str:
    r = httpx.get(VLLM_URL, timeout=10.0)
    r.raise_for_status()
    models = r.json().get("data", [])
    if not models:
        raise RuntimeError("No models loaded in vLLM")
    return models[0]["id"]


def _send_request(client: httpx.Client, model: str, prompt: str, max_tokens: int) -> dict:
    start = time.time()
    resp = client.post(PROXY_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    resp.raise_for_status()
    elapsed = time.time() - start
    usage = resp.json().get("usage", {})
    return {
        "prefill_tokens": usage.get("prompt_tokens", 0),
        "decode_tokens": usage.get("completion_tokens", 0),
        "latency_s": round(elapsed, 3),
    }


def _get_latest_record(client: httpx.Client) -> dict | None:
    """Fetch latest record from proxy stats endpoint."""
    time.sleep(1.5)  # wait for async attribution
    resp = client.get(STATS_URL, timeout=10.0)
    if resp.status_code == 200:
        records = resp.json().get("requests", [])
        if records:
            return records[0]
    return None


def run_benchmark() -> dict:
    try:
        model_id = _detect_model()
    except Exception:
        print("\nERROR: Cannot reach vLLM (port 8000) or proxy (port 8001).")
        print("Start both first, then re-run.")
        sys.exit(1)

    print(f"Model: {model_id}")
    print(f"Test cases: {len(TEST_CASES)} x {ROUNDS_PER_CASE} rounds")
    print()

    client = httpx.Client(timeout=120.0)

    # Warm-up
    print(f"Warm-up ({WARMUP_ROUNDS} requests)...")
    for i in range(WARMUP_ROUNDS):
        _send_request(client, model_id, "hi", 10)
        time.sleep(0.5)
    print("Warm-up done.\n")
    time.sleep(2.0)

    results = []
    for label, prompt, max_tokens in TEST_CASES:
        print(f"--- {label} (max_tokens={max_tokens}) ---")
        case_rounds = []

        for r in range(ROUNDS_PER_CASE):
            time.sleep(1.0)  # settle between rounds
            usage = _send_request(client, model_id, prompt, max_tokens)
            record = _get_latest_record(client)

            energy = record["energy_joules"] if record else 0.0
            cost = record["cost"] if record else 0.0

            round_data = {
                **usage,
                "energy_joules": energy,
                "cost": cost,
            }
            case_rounds.append(round_data)
            print(f"  Round {r+1}: {usage['prefill_tokens']}p+{usage['decode_tokens']}d "
                  f"| {usage['latency_s']:.2f}s | {energy:.4f}J | USD {cost:.10f}")

        # Average across rounds
        avg = {
            "prefill_tokens": round(sum(r["prefill_tokens"] for r in case_rounds) / len(case_rounds)),
            "decode_tokens": round(sum(r["decode_tokens"] for r in case_rounds) / len(case_rounds)),
            "latency_s": round(sum(r["latency_s"] for r in case_rounds) / len(case_rounds), 3),
            "energy_joules": round(sum(r["energy_joules"] for r in case_rounds) / len(case_rounds), 4),
            "cost": sum(r["cost"] for r in case_rounds) / len(case_rounds),
        }
        results.append({"label": label, "avg": avg, "rounds": case_rounds})
        print(f"  AVG: {avg['latency_s']:.2f}s | {avg['energy_joules']:.4f}J | USD {avg['cost']:.10f}")
        print()

    client.close()

    return {"model": model_id, "timestamp": time.time(), "cases": results}


def save_result(result: dict) -> None:
    all_results = []
    if RESULT_PATH.exists():
        all_results = json.loads(RESULT_PATH.read_text())

    # Replace existing entry for same model, or append
    all_results = [r for r in all_results if r["model"] != result["model"]]
    all_results.append(result)

    RESULT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"Results saved to {RESULT_PATH}")


def print_report() -> None:
    if not RESULT_PATH.exists():
        print("No benchmark results found. Run benchmark first.")
        return

    all_results = json.loads(RESULT_PATH.read_text())
    if not all_results:
        print("No benchmark results found.")
        return

    # Collect all case labels
    labels = [c["label"] for c in all_results[0]["cases"]]
    models = [r["model"] for r in all_results]

    print("=" * 80)
    print("PowerDecode Multi-Model Benchmark Report")
    print("=" * 80)
    print(f"Models: {len(models)}")
    print()

    for label in labels:
        print(f"--- {label} ---")
        print(f"  {'Model':<40} {'Latency':>8} {'Energy(J)':>10} {'Cost(USD)':>14} {'P-tok':>6} {'D-tok':>6}")
        for r in all_results:
            case = next((c for c in r["cases"] if c["label"] == label), None)
            if case:
                a = case["avg"]
                print(f"  {r['model']:<40} {a['latency_s']:>7.2f}s {a['energy_joules']:>10.4f} "
                      f"USD {a['cost']:>10.10f} {a['prefill_tokens']:>6} {a['decode_tokens']:>6}")
        print()

    # Summary: total energy across all cases per model
    print("--- Total Energy (all cases) ---")
    for r in all_results:
        total_e = sum(c["avg"]["energy_joules"] for c in r["cases"])
        total_c = sum(c["avg"]["cost"] for c in r["cases"])
        print(f"  {r['model']:<40} {total_e:.4f} J  |  USD {total_c:.10f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    if "--report" in sys.argv:
        print_report()
    else:
        result = run_benchmark()
        save_result(result)
        print()
        print_report()
