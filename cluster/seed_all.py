"""seed_all.py — Full dataset seeding: demo data + middle points + stress test.

Execution order:
  1. warmup (3 requests)
  2. seed_demo_data scenarios (sequential / low-concurrency)
  3. seed_middle_points (sequential)
  4. stress_test Batch 1-5 (concurrent, up to 128)

Usage:
    1. Start vLLM on port 8000
    2. Start proxy.py on port 8001
    3. python3 cluster/seed_all.py
"""

import asyncio
import random
import time

import httpx

PROXY_URL = "http://localhost:8001"
VLLM_URL = "http://localhost:8000"
COMPLETIONS_URL = f"{PROXY_URL}/v1/chat/completions"
STATS_URL = f"{PROXY_URL}/stats/recent"
REQUEST_TIMEOUT = 180.0

MODEL = ""  # auto-detected at startup

# Counters
total_ok = 0
total_failed = 0


# ======================================================================
# Core helpers (from stress_test_h100.py)
# ======================================================================


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


def send_sequential(prompt: str, max_tokens: int, label: str) -> dict:
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


def fetch_recent(n: int) -> list[dict]:
    """GET /stats/recent?limit=N."""
    try:
        resp = httpx.get(f"{STATS_URL}?limit={n}", timeout=60.0)
        resp.raise_for_status()
        return resp.json().get("requests", [])[:n]
    except Exception as e:
        print(f"  ⚠ Failed to fetch stats: {e}")
        return []


# ======================================================================
# Warmup (from stress_test_h100.py)
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
# Reporting
# ======================================================================


def print_stage_header(name: str, count: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({count} requests)")
    print(f"{'='*60}")


def count_results(results: list[dict]) -> tuple[int, int]:
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok")
    failed = len(results) - ok
    print(f"  OK: {ok}  Failed: {failed}")
    return ok, failed


# ======================================================================
# Stage 2: seed_demo_data scenarios
# ======================================================================

TOPICS = [
    "gradient descent", "attention mechanism", "tokenization", "RLHF",
    "KV cache", "tensor parallelism", "batch inference", "quantization",
    "speculative decoding", "flash attention",
]


def original_prefill_requests() -> list[tuple[str, int, str]]:
    return [
        ("Summarize the following text: " + "word " * 400, 50, f"prefill-{i}")
        for i in range(5)
    ]


def original_decode_batch_1() -> list[tuple[str, int, str]]:
    return [
        ("Write a detailed explanation of how transformers work.", 300, f"decode-1-{i}")
        for i in range(5)
    ]


def original_decode_batch_2() -> list[tuple[str, int, str]]:
    return [
        ("Write a detailed explanation of how transformers work.", 300, f"decode-2-{i}")
        for i in range(5)
    ]


def original_mixed_batch_1() -> list[tuple[str, int, str]]:
    return [
        ("Explain the concept of " + random.choice(TOPICS), 150, f"mixed-1-{i}")
        for i in range(5)
    ]


def original_mixed_batch_2() -> list[tuple[str, int, str]]:
    return [
        ("Explain the concept of " + random.choice(TOPICS), 150, f"mixed-2-{i}")
        for i in range(5)
    ]


def batch_a() -> list[tuple[str, int, str]]:
    """極端 decode × 5 — triggers anomaly_flag."""
    return [("Hi.", 500, f"extreme-decode-{i}") for i in range(5)]


def batch_b() -> list[tuple[str, int, str]]:
    """完全相同 × 8 — fair split."""
    return [
        ("Explain what is machine learning in one paragraph.", 150, f"identical-{i}")
        for i in range(8)
    ]


def batch_c() -> list[tuple[str, int, str]]:
    """長 prompt + 長輸出 × 4 — most expensive."""
    return [
        ("word " * 800 + "Write a detailed summary and analysis.", 300, f"expensive-{i}")
        for i in range(4)
    ]


def batch_d() -> list[tuple[str, int, str]]:
    """短問短答 burst × 16."""
    short_prompts = ["What is 2+2?", "Name a color.", "Say hello.", "What day is today?"]
    return [
        (random.choice(short_prompts), 10, f"burst-{i}")
        for i in range(16)
    ]


def batch_e() -> list[tuple[str, int, str]]:
    """8 similar-latency requests."""
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


async def stage_seed_demo() -> None:
    """Run all seed_demo_data scenarios."""
    global total_ok, total_failed

    # Prefill-heavy: sequential
    print_stage_header("Demo — Prefill-heavy (sequential)", 5)
    for prompt, mt, label in original_prefill_requests():
        result = send_sequential(prompt, mt, label)
        if result["status"] == "ok":
            total_ok += 1
            print(f"  {label}: ok")
        else:
            total_failed += 1
            print(f"  {label}: failed — {result.get('error', '')}")
        await asyncio.sleep(1.5)

    # Concurrent batches
    demo_batches = [
        ("Demo — Decode-heavy batch 1", original_decode_batch_1()),
        ("Demo — Decode-heavy batch 2", original_decode_batch_2()),
        ("Demo — Mixed batch 1", original_mixed_batch_1()),
        ("Demo — Mixed batch 2", original_mixed_batch_2()),
    ]
    for name, reqs in demo_batches:
        print_stage_header(name, len(reqs))
        results = await run_batch(reqs)
        ok, failed = count_results(results)
        total_ok += ok
        total_failed += failed
        await asyncio.sleep(2.0)

    # Batch A — sequential (needs baseline for anomaly detection)
    print_stage_header("Demo — BATCH A: Extreme decode (sequential)", 5)
    for prompt, mt, label in batch_a():
        result = send_sequential(prompt, mt, label)
        if result["status"] == "ok":
            total_ok += 1
            print(f"  {label}: ok")
        else:
            total_failed += 1
            print(f"  {label}: failed — {result.get('error', '')}")
        await asyncio.sleep(2.0)

    # Batches B-E — concurrent
    demo_batches_2 = [
        ("Demo — BATCH B: Identical (fair split)", batch_b()),
        ("Demo — BATCH C: Expensive (long prompt+output)", batch_c()),
        ("Demo — BATCH D: Short burst", batch_d()),
        ("Demo — BATCH E: Similar-latency", batch_e()),
    ]
    for name, reqs in demo_batches_2:
        print_stage_header(name, len(reqs))
        results = await run_batch(reqs)
        ok, failed = count_results(results)
        total_ok += ok
        total_failed += failed
        await asyncio.sleep(2.0)


# ======================================================================
# Stage 3: seed_middle_points
# ======================================================================

MIDDLE_REQUESTS = [
    # 極低 decode 比例（decode token % < 10%）
    ("word " * 200 + "What is the answer?", 8),
    ("word " * 300 + "Summarize in one word.", 5),
    ("word " * 400 + "Give a one-word answer.", 6),
    ("word " * 150 + "Yes or no?", 3),
    ("word " * 250 + "Name one thing.", 5),
    ("word " * 500 + "One word only.", 4),
    ("word " * 350 + "True or false?", 4),
    # 低 decode 比例（decode token % 15-30%）
    ("word " * 100 + "Explain in 20 words.", 20),
    ("word " * 80 + "Summarize briefly.", 25),
    ("word " * 120 + "Describe in one sentence.", 18),
    ("word " * 90 + "What is the key point?", 22),
    ("word " * 70 + "Give a short explanation.", 20),
    ("word " * 110 + "List two key facts.", 18),
    ("word " * 130 + "Answer concisely.", 22),
    ("word " * 60 + "What does this mean?", 15),
    ("word " * 85 + "Explain the main idea.", 20),
    ("word " * 140 + "Respond in two sentences.", 25),
    ("word " * 95 + "State the conclusion.", 18),
    ("word " * 75 + "Why is this important?", 22),
    # 中低比例（decode token % 30-40%）
    ("word " * 50 + "Explain in 30 words.", 30),
    ("word " * 40 + "Summarize the concept.", 25),
    ("word " * 55 + "Describe the process.", 32),
    ("word " * 45 + "What are the key points?", 28),
    ("word " * 60 + "Give a brief overview.", 35),
    ("word " * 35 + "Explain step by step.", 22),
    ("word " * 50 + "What is the significance?", 30),
    ("word " * 42 + "Describe the approach.", 26),
    ("word " * 48 + "Outline the method.", 28),
    ("word " * 55 + "What are the benefits?", 32),
    ("word " * 38 + "Explain the tradeoffs.", 24),
    ("word " * 52 + "How does this compare?", 30),
    # 中間比例（decode token % 40-60%）
    ("Explain AI in 50 words.", 50),
    ("What is machine learning?", 55),
    ("Describe a GPU briefly.", 48),
    ("What is deep learning?", 52),
    ("Explain transformers briefly.", 50),
    ("What is backpropagation?", 46),
    ("Define neural network.", 50),
    ("What is overfitting?", 54),
    ("Explain gradient descent.", 48),
    ("What is a tensor?", 50),
    ("How does attention work?", 52),
    ("What is a learning rate?", 48),
    # 稍高比例（decode token % 60-75%）
    ("Tell me about AI.", 100),
    ("Explain inference in LLMs.", 90),
    ("What is CUDA?", 85),
    ("How does vLLM work?", 95),
    ("Explain KV cache.", 88),
]


async def stage_middle_points() -> None:
    """Run seed_middle_points sequentially."""
    global total_ok, total_failed

    print_stage_header("Middle points (sequential)", len(MIDDLE_REQUESTS))
    async with httpx.AsyncClient() as client:
        for prompt, max_tokens in MIDDLE_REQUESTS:
            result = await send_request(client, prompt, max_tokens, label="middle")
            if isinstance(result, dict) and result.get("status") == "ok":
                total_ok += 1
                p = result.get("prefill_tokens", 0)
                d = result.get("decode_tokens", 0)
                pct = d / (p + d) * 100 if (p + d) > 0 else 0
                print(f"  ok | decode {pct:.0f}% | {prompt[:45]}")
            else:
                total_failed += 1
                print(f"  failed | {prompt[:45]}")
            await asyncio.sleep(0.3)


# ======================================================================
# Stage 4: stress_test batches 1-5 (from stress_test_h100.py)
# ======================================================================


def stress_batch_1() -> list[tuple[str, int, str]]:
    """Baseline: 2 identical requests."""
    prompt = "Explain what is a GPU."
    return [(prompt, 300, "identical-A"), (prompt, 300, "identical-B")]


def stress_batch_2() -> list[tuple[str, int, str]]:
    """Weight verification: prefill-heavy vs decode-heavy."""
    return [
        ("word " * 800 + "Summarize.", 20, "prefill-heavy"),
        ("Tell me a story.", 600, "decode-heavy"),
    ]


def stress_batch_3() -> list[tuple[str, int, str]]:
    """Medium concurrency: 16 requests."""
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


def stress_batch_4() -> list[tuple[str, int, str]]:
    """High concurrency: 64 requests."""
    reqs = []
    for i in range(32):
        reqs.append(("What is 2+2?", 50, f"short-{i}"))
    for i in range(32):
        reqs.append(("Write a detailed essay about AI.", 500, f"long-{i}"))
    return reqs


def stress_batch_5() -> list[tuple[str, int, str]]:
    """Stress test: 128 concurrent requests."""
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


STRESS_BATCHES = [
    ("Stress BATCH 1 — Baseline (identical)", stress_batch_1),
    ("Stress BATCH 2 — Weight verification", stress_batch_2),
    ("Stress BATCH 3 — Medium concurrency (16)", stress_batch_3),
    ("Stress BATCH 4 — High concurrency (64)", stress_batch_4),
    ("Stress BATCH 5 — Stress test (128)", stress_batch_5),
]


async def stage_stress_test() -> None:
    """Run stress_test batches 1-5 concurrently."""
    global total_ok, total_failed

    for name, build_fn in STRESS_BATCHES:
        reqs = build_fn()
        print_stage_header(name, len(reqs))

        t0 = time.time()
        results = await run_batch(reqs)
        elapsed = time.time() - t0

        ok, failed = count_results(results)
        total_ok += ok
        total_failed += failed
        print(f"  Batch wall time: {elapsed:.1f}s")

        if name != STRESS_BATCHES[-1][0]:
            await asyncio.sleep(2.0)


# ======================================================================
# Main
# ======================================================================


async def main() -> None:
    global MODEL
    print("Detecting model from vLLM...")
    MODEL = detect_model()
    print(f"  Model: {MODEL}\n")

    # ① Warmup
    await warmup()

    # ② seed_demo_data scenarios
    print(f"\n{'#'*60}")
    print("  STAGE 2: seed_demo_data scenarios")
    print(f"{'#'*60}")
    await stage_seed_demo()

    await asyncio.sleep(2.0)

    # ③ seed_middle_points
    print(f"\n{'#'*60}")
    print("  STAGE 3: seed_middle_points")
    print(f"{'#'*60}")
    await stage_middle_points()

    await asyncio.sleep(2.0)

    # ④ stress_test Batch 1-5
    print(f"\n{'#'*60}")
    print("  STAGE 4: stress_test Batch 1-5")
    print(f"{'#'*60}")
    await stage_stress_test()

    # ==================================================================
    # Final summary
    # ==================================================================
    await asyncio.sleep(2.0)
    all_records = fetch_recent(500)
    total_count = len(all_records)
    anomaly_1 = sum(1 for r in all_records if r.get("anomaly_flag") == 1)
    anomaly_2 = sum(1 for r in all_records if r.get("anomaly_flag") == 2)
    costs = [r.get("cost", 0) or 0 for r in all_records]
    max_cost = max(costs) if costs else 0
    min_cost = min(costs) if costs else 0

    print(f"\n{'='*60}")
    print("  === seed_all 完成 ===")
    print(f"  總 request 數：{total_count}")
    print(f"  成功 / 失敗：{total_ok} / {total_failed}")
    print(f"  anomaly_flag=1：{anomaly_1}")
    print(f"  anomaly_flag=2：{anomaly_2}")
    print(f"  最高單筆成本：${max_cost:.8f}")
    print(f"  最低單筆成本：${min_cost:.8f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
