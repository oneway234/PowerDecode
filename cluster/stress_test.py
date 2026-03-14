"""PowerDecode Concurrent Stress Test.

Tests attribution correctness under high concurrency (up to 64 requests).
Sends 5 batches of increasing parallelism through the proxy,
then fetches /stats/recent to verify cost distribution.

Usage:
    1. Start vLLM on port 8000
    2. Start proxy.py on port 8001
    3. python3 cluster/stress_test.py
"""

import asyncio
import random
import time

import httpx

PROXY_URL = "http://localhost:8001"
VLLM_URL = "http://localhost:8000"
MODEL = ""  # auto-detected at startup


def detect_model() -> str:
    """Auto-detect model from vLLM /v1/models."""
    try:
        resp = httpx.get(f"{VLLM_URL}/v1/models", timeout=10.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        print(f"  Warning: could not detect model: {e}")
    return "Qwen/Qwen2.5-3B-Instruct"
COMPLETIONS_URL = f"{PROXY_URL}/v1/chat/completions"
STATS_URL = f"{PROXY_URL}/stats/recent"
REQUEST_TIMEOUT = 120.0


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
    try:
        resp = httpx.get(STATS_URL, timeout=60.0)
        resp.raise_for_status()
        records = resp.json().get("requests", [])
        return records[:n]
    except Exception as e:
        print(f"  ⚠ Failed to fetch stats: {e}")
        return []


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
# Batch definitions
# ======================================================================


def batch_1() -> list[tuple[str, int, str]]:
    """Baseline: 2 identical requests, expect ~50/50 cost split."""
    prompt = "Explain what is a GPU."
    return [
        (prompt, 100, "identical-A"),
        (prompt, 100, "identical-B"),
    ]


def batch_2() -> list[tuple[str, int, str]]:
    """Weight verification: prefill-heavy vs decode-heavy."""
    return [
        ("word " * 500 + "Summarize the above.", 20, "prefill-heavy"),
        ("Tell me a story.", 300, "decode-heavy"),
    ]


def batch_3() -> list[tuple[str, int, str]]:
    """Medium concurrency: 8 requests, mixed prefill/decode."""
    reqs = []
    for i in range(4):
        reqs.append(("word " * 300 + "What is the main idea?", 20, f"prefill-{i}"))
    for i in range(4):
        reqs.append(("Explain transformer architecture in detail.", 200, f"decode-{i}"))
    # Interleave
    interleaved = []
    for i in range(4):
        interleaved.append(reqs[i])
        interleaved.append(reqs[i + 4])
    return interleaved


def batch_4() -> list[tuple[str, int, str]]:
    """High concurrency: 32 requests."""
    reqs = []
    for i in range(16):
        reqs.append(("What is 2+2?", 10, f"short-{i}"))
    for i in range(16):
        reqs.append(("Write a detailed essay about AI.", 300, f"long-{i}"))
    return reqs


def batch_5() -> list[tuple[str, int, str]]:
    """Stress test: 64 requests."""
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
    for i in range(32):
        p = prompts_mixed[i % len(prompts_mixed)]
        mt = random.randint(50, 300)
        reqs.append((p, mt, f"mixed-{i}"))
    for i in range(32):
        reqs.append(("Say hello.", 10, f"short-{i}"))
    return reqs


# ======================================================================
# Main
# ======================================================================

BATCHES = [
    ("BATCH 1 — Baseline (identical requests)", batch_1),
    ("BATCH 2 — Weight verification (prefill vs decode)", batch_2),
    ("BATCH 3 — Medium concurrency", batch_3),
    ("BATCH 4 — High concurrency", batch_4),
    ("BATCH 5 — Stress test", batch_5),
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
        wait_time = 2.0 if len(reqs) <= 8 else 5.0
        print(f"  Waiting {wait_time}s for attribution...")
        await asyncio.sleep(wait_time)

        stats = fetch_recent_stats(len(reqs))
        print_attribution_table(stats)

        # Inter-batch cooldown
        if name != BATCHES[-1][0]:
            print("\n  Cooling down 2s before next batch...")
            await asyncio.sleep(2.0)

    print(f"\n{'='*60}")
    print("  Stress test complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
