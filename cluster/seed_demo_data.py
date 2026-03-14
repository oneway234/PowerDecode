"""Seed demo data — send requests to proxy so the dashboard looks convincing.

Covers all three dashboard pages: Overview, Request Detail, Cost Trend.
Max concurrency: 32.  Total: ~66 requests.

Usage:
    1. Start vLLM + proxy (./start.sh or manually)
    2. python3 cluster/seed_demo_data.py
"""

import asyncio
import random
import time

import httpx

PROXY_URL = "http://localhost:8001"
VLLM_URL = "http://localhost:8000"
COMPLETIONS_URL = f"{PROXY_URL}/v1/chat/completions"
STATS_URL = f"{PROXY_URL}/stats/recent"
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
REQUEST_TIMEOUT = 120.0

TOPICS = [
    "gradient descent",
    "attention mechanism",
    "tokenization",
    "RLHF",
    "KV cache",
    "tensor parallelism",
    "batch inference",
    "quantization",
    "speculative decoding",
    "flash attention",
]

# ======================================================================
# Helpers
# ======================================================================


async def send_one(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    label: str = "",
) -> dict:
    """Send a single chat completion. Returns result dict."""
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
    except Exception as e:
        return {
            "label": label,
            "status": "failed",
            "elapsed": round(time.time() - t0, 2),
            "error": str(e),
        }


async def send_batch(
    requests: list[tuple[str, int, str]],
) -> list[dict]:
    """Send all requests concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [send_one(client, p, m, l) for p, m, l in requests]
        return await asyncio.gather(*tasks)


def fetch_recent(n: int) -> list[dict]:
    """GET /stats/recent?limit=N. Returns up to n records."""
    try:
        resp = httpx.get(f"{STATS_URL}?limit={n}", timeout=60.0)
        resp.raise_for_status()
        return resp.json().get("requests", [])[:n]
    except Exception as e:
        print(f"  ⚠ Failed to fetch stats: {e}")
        return []


def send_sequential(
    prompt: str, max_tokens: int, label: str
) -> dict:
    """Send a single request synchronously (no concurrency)."""
    try:
        resp = httpx.post(
            COMPLETIONS_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
        return {"label": label, "status": "ok",
                "prefill_tokens": usage.get("prompt_tokens", 0),
                "decode_tokens": usage.get("completion_tokens", 0)}
    except Exception as e:
        return {"label": label, "status": "failed", "error": str(e)}


# ======================================================================
# Reporting
# ======================================================================


def print_batch_header(name: str, count: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({count} requests)")
    print(f"{'='*60}")


def print_results(results: list[dict]) -> None:
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"  OK: {ok}  Failed: {failed}")


def print_attribution(stats: list[dict], highlight_label: str | None = None) -> None:
    if not stats:
        print("  (no attribution records)")
        return
    total_energy = sum(r.get("energy_joules", 0) or 0 for r in stats)
    for r in stats:
        energy = r.get("energy_joules", 0) or 0
        cost = r.get("cost", 0) or 0
        preview = (r.get("prompt_preview", "") or "")[:38]
        share = f"{energy / total_energy * 100:.1f}%" if total_energy > 0 else "N/A"
        anomaly = " ← ANOMALY" if r.get("anomaly_flag") else ""
        print(f"  {preview:<40} {energy:>10.4f}J  USD {cost:.8f}  {share:>6}{anomaly}")


# ======================================================================
# Warmup
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
# Original batches (adjusted concurrency)
# ======================================================================


def original_prefill_requests() -> list[tuple[str, int, str]]:
    """5 prefill-heavy, sent sequentially."""
    return [
        ("Summarize the following text: " + "word " * 400, 50, f"prefill-{i}")
        for i in range(5)
    ]


def original_decode_batch_1() -> list[tuple[str, int, str]]:
    """5 decode-heavy, concurrent."""
    return [
        ("Write a detailed explanation of how transformers work.", 300, f"decode-1-{i}")
        for i in range(5)
    ]


def original_decode_batch_2() -> list[tuple[str, int, str]]:
    """5 decode-heavy, concurrent."""
    return [
        ("Write a detailed explanation of how transformers work.", 300, f"decode-2-{i}")
        for i in range(5)
    ]


def original_mixed_batch_1() -> list[tuple[str, int, str]]:
    """5 mixed, concurrent."""
    return [
        ("Explain the concept of " + random.choice(TOPICS), 150, f"mixed-1-{i}")
        for i in range(5)
    ]


def original_mixed_batch_2() -> list[tuple[str, int, str]]:
    """5 mixed, concurrent."""
    return [
        ("Explain the concept of " + random.choice(TOPICS), 150, f"mixed-2-{i}")
        for i in range(5)
    ]


# ======================================================================
# New batches A-E
# ======================================================================


def batch_a() -> list[tuple[str, int, str]]:
    """BATCH A — 極端 decode × 5 (concurrent 5).
    Triggers anomaly_flag. Donut shows 99% decode.
    Uses very short prompt + high max_tokens so decode dominates.
    Sent one-by-one to avoid energy splitting (each gets full GPU power).
    """
    return [("Hi.", 500, f"extreme-decode-{i}") for i in range(5)]


def batch_b() -> list[tuple[str, int, str]]:
    """BATCH B — 完全相同 × 8 (concurrent 8).
    Shows fair split, each ~12.5%.
    """
    return [
        ("Explain what is machine learning in one paragraph.", 150, f"identical-{i}")
        for i in range(8)
    ]


def batch_c() -> list[tuple[str, int, str]]:
    """BATCH C — 長 prompt + 長輸出 × 4 (concurrent 4).
    Cost scatter top-right. Most expensive requests.
    """
    return [
        ("word " * 800 + "Write a detailed summary and analysis.", 300, f"expensive-{i}")
        for i in range(4)
    ]


def batch_d() -> list[tuple[str, int, str]]:
    """BATCH D — 短問短答 burst × 16 (concurrent 16).
    Cost trend gets a dense staircase segment.
    """
    short_prompts = [
        "What is 2+2?",
        "Name a color.",
        "Say hello.",
        "What day is today?",
    ]
    return [
        (random.choice(short_prompts), 10, f"burst-{i}")
        for i in range(16)
    ]


def batch_e() -> list[tuple[str, int, str]]:
    """BATCH E — 8 similar-latency requests (concurrent 8).
    All use max_tokens=150 with similar-length prompts for fair attribution.
    """
    prompts = [
        "Explain what is machine learning.",
        "Explain what is deep learning.",
        "Explain what is a neural network.",
        "Explain what is backpropagation.",
        "Explain what is gradient descent.",
        "Explain what is overfitting.",
        "Explain what is a transformer.",
        "Explain what is attention mechanism.",
    ]
    return [(p, 150, f"similar-{i}") for i, p in enumerate(prompts)]



# ======================================================================
# Main
# ======================================================================


async def main() -> None:
    global MODEL
    print("Detecting model from vLLM...")
    MODEL = detect_model()
    print(f"  Model: {MODEL}\n")

    await warmup()

    all_stats: list[dict] = []  # collect all stats records
    total_ok = 0
    total_failed = 0

    # ------------------------------------------------------------------
    # Original batches
    # ------------------------------------------------------------------

    # Prefill-heavy: sequential (no concurrency)
    print_batch_header("Original — Prefill-heavy (sequential)", 5)
    prefill_reqs = original_prefill_requests()
    for prompt, mt, label in prefill_reqs:
        result = send_sequential(prompt, mt, label)
        if result["status"] == "ok":
            total_ok += 1
            print(f"  {label}: ok")
        else:
            total_failed += 1
            print(f"  {label}: failed — {result.get('error', '')}")
        await asyncio.sleep(1.5)
    await asyncio.sleep(2.0)
    stats = fetch_recent(5)
    print_attribution(stats)

    # Decode-heavy batch 1
    print_batch_header("Original — Decode-heavy batch 1", 5)
    results = await send_batch(original_decode_batch_1())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(3.0)
    stats = fetch_recent(5)
    print_attribution(stats)

    # Decode-heavy batch 2
    print_batch_header("Original — Decode-heavy batch 2", 5)
    results = await send_batch(original_decode_batch_2())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(3.0)
    stats = fetch_recent(5)
    print_attribution(stats)

    # Mixed batch 1
    print_batch_header("Original — Mixed batch 1", 5)
    results = await send_batch(original_mixed_batch_1())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(3.0)
    stats = fetch_recent(5)
    print_attribution(stats)

    # Mixed batch 2
    print_batch_header("Original — Mixed batch 2", 5)
    results = await send_batch(original_mixed_batch_2())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(3.0)
    stats = fetch_recent(5)
    print_attribution(stats)

    print(f"\n  Original batches done: {total_ok} ok / {total_failed} failed")
    print(f"  (Baseline established: {total_ok} records for anomaly detection)")
    await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # BATCH A — 極端 decode (sequential, needs baseline for anomaly)
    # ------------------------------------------------------------------
    print_batch_header("BATCH A — Extreme decode (anomaly trigger, sequential)", 5)
    batch_a_reqs = batch_a()
    for prompt, mt, label in batch_a_reqs:
        result = send_sequential(prompt, mt, label)
        if result["status"] == "ok":
            total_ok += 1
            print(f"  {label}: ok")
        else:
            total_failed += 1
            print(f"  {label}: failed — {result.get('error', '')}")
        await asyncio.sleep(2.0)
    await asyncio.sleep(3.0)
    stats_a = fetch_recent(5)
    print_attribution(stats_a)
    anomalies_a = sum(1 for r in stats_a if r.get("anomaly_flag"))
    print(f"  Anomalies triggered: {anomalies_a}")

    await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # BATCH B — 完全相同 × 8 (fair split)
    # ------------------------------------------------------------------
    print_batch_header("BATCH B — Identical requests (fair split)", 8)
    results = await send_batch(batch_b())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(5.0)
    stats_b = fetch_recent(8)
    print_attribution(stats_b)
    if stats_b:
        total_energy_b = sum(r.get("energy_joules", 0) or 0 for r in stats_b)
        if total_energy_b > 0:
            shares = [(r.get("energy_joules", 0) or 0) / total_energy_b * 100 for r in stats_b]
            in_range = sum(1 for s in shares if 10 <= s <= 15)
            print(f"  Shares in 10-15% range: {in_range}/{len(shares)}")

    await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # BATCH C — 長 prompt + 長輸出 (most expensive)
    # ------------------------------------------------------------------
    print_batch_header("BATCH C — Long prompt + long output (expensive)", 4)
    results = await send_batch(batch_c())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(5.0)
    stats_c = fetch_recent(4)
    print_attribution(stats_c)

    await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # BATCH D — 短問短答 burst × 16
    # ------------------------------------------------------------------
    print_batch_header("BATCH D — Short burst (cost trend staircase)", 16)
    results = await send_batch(batch_d())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(5.0)
    stats_d = fetch_recent(16)
    print_attribution(stats_d)

    await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # BATCH E — 8 similar-latency requests
    # ------------------------------------------------------------------
    print_batch_header("BATCH E — 8 similar-latency requests (fair split)", 8)
    results = await send_batch(batch_e())
    print_results(results)
    total_ok += sum(1 for r in results if r["status"] == "ok")
    total_failed += sum(1 for r in results if r["status"] == "failed")
    await asyncio.sleep(5.0)
    stats_e = fetch_recent(8)
    print_attribution(stats_e)

    # ==================================================================
    # Final summary
    # ==================================================================
    await asyncio.sleep(2.0)

    # Fetch all records to compute summary
    all_records = fetch_recent(200)
    total_count = len(all_records)
    anomaly_count = sum(1 for r in all_records if r.get("anomaly_flag") == 1)
    extreme_count_all = sum(1 for r in all_records if r.get("anomaly_flag") == 2)
    costs = [r.get("cost", 0) or 0 for r in all_records]
    energies_all = [r.get("energy_joules", 0) or 0 for r in all_records]
    max_cost = max(costs) if costs else 0
    min_cost = min(costs) if costs else 0

    # Check dashboard criteria
    has_anomaly = anomaly_count > 0
    has_decode_heavy = any(
        (r.get("decode_tokens", 0) or 0) > 9 * max((r.get("prefill_tokens", 0) or 0), 1)
        for r in all_records
    )
    has_prefill_heavy = any(
        (r.get("prefill_tokens", 0) or 0) > 1.5 * max((r.get("decode_tokens", 0) or 0), 1)
        for r in all_records
    )
    has_slope_change = total_count >= 30  # enough data for visible trend change

    check = lambda ok: "✓" if ok else "✗"

    print(f"\n{'='*60}")
    print("  seed_demo_data 完成")
    print(f"{'='*60}")
    print(f"  總 request 數：{total_count}")
    print(f"  成功 / 失敗：{total_ok} / {total_failed}")
    print(f"  anomaly_flag=1 的筆數：{anomaly_count}（目標 ≥ 2）")
    print(f"  anomaly_flag=2 (extreme) 的筆數：{extreme_count_all}")
    print(f"  最高單筆成本：USD {max_cost:.8f}")
    print(f"  最低單筆成本：USD {min_cost:.8f}")
    print()
    print("  Dashboard 檢查清單：")
    print(f"  {check(has_anomaly)} Overview：有標紅 request")
    print(f"  {check(has_decode_heavy)} Request Detail：有 decode > 90% 的 request")
    print(f"  {check(has_prefill_heavy)} Request Detail：有 prefill > 60% 的 request")
    print(f"  {check(has_slope_change)} Cost Trend：有明顯斜率變化")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
