"""PowerDecode Concurrent Stress Test — Fluidstack H100 Edition.

Based on stress_test.py but tuned for H100 characteristics:
  - Higher decode throughput (~600 tok/s) → longer max_tokens for overlap
  - Larger batch capacity → up to 128 concurrent
  - Faster prefill (<100ms) → longer prompts for sufficient sampling
  - Higher idle power (~50W)

Auto-detects model from vLLM /v1/models.
After all batches, prints H100 vs 4060 Ti comparison from benchmark_baselines.json.

Usage:
    1. Start vLLM on port 8000
    2. Start proxy.py on port 8001
    3. python3 cluster/stress_test_h100.py
"""

import asyncio
import json
import random
import time
from pathlib import Path

import httpx

PROXY_URL = "http://localhost:8001"
VLLM_URL = "http://localhost:8000"
COMPLETIONS_URL = f"{PROXY_URL}/v1/chat/completions"
STATS_URL = f"{PROXY_URL}/stats/recent"
REQUEST_TIMEOUT = 180.0
BASELINES_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_baselines.json"

# Auto-detected at startup
MODEL = ""


def detect_model() -> str:
    """Auto-detect model from vLLM /v1/models."""
    resp = httpx.get(f"{VLLM_URL}/v1/models", timeout=10.0)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError("No models loaded in vLLM")
    return models[0]["id"]


# ======================================================================
# Request helpers
# ======================================================================


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    label: str = "",
) -> dict:
    """Send a single chat completion request. Returns result dict."""
    t0 = time.time()
    try:
        resp = await client.post(
            COMPLETIONS_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        elapsed = time.time() - t0
        usage = resp.json().get("usage", {})
        return {
            "label": label,
            "status": "ok",
            "elapsed": round(elapsed, 2),
            "prefill_tokens": usage.get("prompt_tokens", 0),
            "decode_tokens": usage.get("completion_tokens", 0),
        }
    except httpx.TimeoutException:
        return {"label": label, "status": "timeout", "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"label": label, "status": "failed", "elapsed": round(time.time() - t0, 2), "error": str(e)}


async def run_batch(requests_list: list[tuple[str, int, str]]) -> list[dict]:
    """Send all requests concurrently and return results."""
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, p, m, l) for p, m, l in requests_list]
        return await asyncio.gather(*tasks, return_exceptions=True)


def fetch_recent_stats(n: int) -> list[dict]:
    """GET /stats/recent and return the latest n records."""
    resp = httpx.get(STATS_URL, timeout=10.0)
    resp.raise_for_status()
    records = resp.json().get("requests", [])
    return records[:n]


# ======================================================================
# Reporting
# ======================================================================


def print_batch_header(name: str, count: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({count} concurrent)")
    print(f"{'='*60}")


def print_request_results(results: list[dict]) -> None:
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok")
    failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    timeout = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "timeout")
    errors = sum(1 for r in results if isinstance(r, Exception))
    print(f"  Requests: {ok} ok / {failed} failed / {timeout} timeout / {errors} exception")


def print_attribution_table(stats: list[dict]) -> None:
    if not stats:
        print("  (no attribution records found)")
        return

    print(f"  {'Prompt':<40} {'Energy(J)':>10} {'Cost(USD)':>10} {'Share':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")

    total_energy = sum(r.get("energy_joules", 0) or 0 for r in stats)
    for r in stats:
        energy = r.get("energy_joules", 0) or 0
        cost = r.get("cost", 0) or 0
        preview = (r.get("prompt_preview", "") or "")[:38]
        share = f"{energy / total_energy * 100:.1f}%" if total_energy > 0 else "N/A"
        print(f"  {preview:<40} {energy:>10.4f} {cost:>10.6f} {share:>8}")

    print(f"  {'TOTAL':<40} {total_energy:>10.4f}")


# ======================================================================
# Warm-up
# ======================================================================


async def warmup() -> None:
    print("Warming up (3 requests)...")
    async with httpx.AsyncClient() as client:
        for i in range(3):
            try:
                await client.post(
                    COMPLETIONS_URL,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 5,
                    },
                    timeout=REQUEST_TIMEOUT,
                    headers={"X-Warmup": "true"},
                )
                print(f"  warm-up {i+1}/3 done")
            except Exception as e:
                print(f"  warm-up {i+1}/3 failed: {e}")
            await asyncio.sleep(0.5)
    print("Warm-up complete.\n")


# ======================================================================
# Batch definitions (H100-tuned)
# ======================================================================


def batch_1() -> list[tuple[str, int, str]]:
    """Baseline: 2 identical requests, expect ~50/50 cost split.

    max_tokens=300 (up from 100) to ensure overlap on fast H100 decode.
    """
    prompt = "Explain what is a GPU."
    return [
        (prompt, 300, "identical-A"),
        (prompt, 300, "identical-B"),
    ]


def batch_2() -> list[tuple[str, int, str]]:
    """Weight verification: prefill-heavy vs decode-heavy.

    Prefill prompt = 1600 words to ensure H100 prefill has enough sampling points.
    Decode max_tokens = 600 for sufficient decode duration on H100.
    """
    return [
        ("word " * 1600 + "Summarize.", 20, "prefill-heavy"),
        ("Tell me a story.", 600, "decode-heavy"),
    ]


def batch_3() -> list[tuple[str, int, str]]:
    """Medium concurrency: 16 requests (up from 8 for H100 capacity).

    8 prefill-heavy + 8 decode-heavy, interleaved.
    """
    reqs = []
    for i in range(8):
        reqs.append(("word " * 800 + "What is the main idea?", 20, f"prefill-{i}"))
    for i in range(8):
        reqs.append(("Explain transformer architecture in detail.", 400, f"decode-{i}"))
    interleaved = []
    for i in range(8):
        interleaved.append(reqs[i])
        interleaved.append(reqs[i + 8])
    return interleaved


def batch_4() -> list[tuple[str, int, str]]:
    """High concurrency: 64 requests (up from 32 for H100 capacity)."""
    reqs = []
    for i in range(32):
        reqs.append(("What is 2+2?", 50, f"short-{i}"))
    for i in range(32):
        reqs.append(("Write a detailed essay about AI.", 500, f"long-{i}"))
    return reqs


def batch_5() -> list[tuple[str, int, str]]:
    """H100 stress test: 128 concurrent requests."""
    reqs = []
    prompts_mixed = [
        "Explain how neural networks learn.",
        "What are the benefits of renewable energy?",
        "Describe the water cycle.",
        "How does a compiler work?",
        "What is quantum computing?",
        "Explain the theory of relativity.",
        "How do databases handle transactions?",
        "What is the difference between TCP and UDP?",
    ]
    for i in range(64):
        p = prompts_mixed[i % len(prompts_mixed)]
        mt = random.randint(100, 600)
        reqs.append((p, mt, f"mixed-{i}"))
    for i in range(64):
        reqs.append(("Say hello.", 50, f"short-{i}"))
    return reqs


# ======================================================================
# H100 vs 4060 Ti comparison
# ======================================================================


def print_gpu_comparison() -> None:
    """Load benchmark_baselines.json and print H100 vs 4060 Ti side-by-side."""
    print(f"\n{'='*60}")
    print("  H100 vs 4060 Ti — Baseline Comparison")
    print(f"{'='*60}")

    if not BASELINES_PATH.exists():
        print("  benchmark_baselines.json not found — skipping comparison.")
        return

    data = json.loads(BASELINES_PATH.read_text())
    baselines = data.get("baselines", [])

    # Find matching entries (same model on both GPUs)
    by_key: dict[str, dict[str, dict]] = {}
    for b in baselines:
        model = b["model"]
        gpu = b["gpu"]
        by_key.setdefault(model, {})[gpu] = b

    has_data = False
    for model, gpus in by_key.items():
        h100 = gpus.get("NVIDIA H100")
        ti = gpus.get("RTX 4060 Ti 16GB")
        if not h100 and not ti:
            continue

        print(f"\n  Model: {model}")
        print(f"  {'Metric':<25} {'4060 Ti':>12} {'H100':>12} {'Ratio':>8}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")

        ti_idle = ti.get("idle_power_w") if ti else None
        h100_idle = h100.get("idle_power_w") if h100 else None
        ti_wp = ti.get("w_prefill") if ti else None
        h100_wp = h100.get("w_prefill") if h100 else None
        ti_wd = ti.get("w_decode") if ti else None
        h100_wd = h100.get("w_decode") if h100 else None

        def _fmt(v: float | None) -> str:
            if v is None:
                return "pending"
            return f"{v:.4f}"

        def _ratio(a: float | None, b: float | None) -> str:
            if a is None or b is None or b == 0:
                return "-"
            return f"{a / b:.2f}x"

        print(f"  {'Idle power (W)':<25} {_fmt(ti_idle):>12} {_fmt(h100_idle):>12} {_ratio(h100_idle, ti_idle):>8}")
        print(f"  {'W_prefill':<25} {_fmt(ti_wp):>12} {_fmt(h100_wp):>12} {_ratio(h100_wp, ti_wp):>8}")
        print(f"  {'W_decode':<25} {_fmt(ti_wd):>12} {_fmt(h100_wd):>12} {_ratio(h100_wd, ti_wd):>8}")

        if h100_idle is None or h100_wp is None or h100_wd is None:
            print("\n  ⚠ H100 baseline pending — run calibrate.py first")
        else:
            has_data = True

    if not has_data:
        print("\n  No complete H100 data yet. Run calibrate.py on H100 to populate.")


# ======================================================================
# Main
# ======================================================================

BATCHES = [
    ("BATCH 1 — Baseline (identical requests)", batch_1),
    ("BATCH 2 — Weight verification (prefill vs decode)", batch_2),
    ("BATCH 3 — Medium concurrency (H100)", batch_3),
    ("BATCH 4 — High concurrency (H100)", batch_4),
    ("BATCH 5 — H100 stress test", batch_5),
]


async def main() -> None:
    global MODEL
    print("Detecting model from vLLM...")
    MODEL = detect_model()
    print(f"  Model: {MODEL}\n")

    await warmup()

    for name, build_fn in BATCHES:
        reqs = build_fn()
        print_batch_header(name, len(reqs))

        t0 = time.time()
        results = await run_batch(reqs)
        elapsed = time.time() - t0

        print_request_results(results)
        print(f"  Batch wall time: {elapsed:.1f}s")

        # Wait for attribution engine to finish processing
        print("  Waiting 2s for attribution...")
        await asyncio.sleep(2.0)

        stats = fetch_recent_stats(len(reqs))
        print_attribution_table(stats)

        # Inter-batch cooldown
        if name != BATCHES[-1][0]:
            print("\n  Cooling down 2s before next batch...")
            await asyncio.sleep(2.0)

    # GPU comparison summary
    print_gpu_comparison()

    print(f"\n{'='*60}")
    print("  H100 stress test complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
