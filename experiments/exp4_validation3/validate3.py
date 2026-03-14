"""Validation 3: Batch-level energy conservation.

Sends three concurrent requests with different max_tokens (200, 400, 600),
then verifies that the sum of attributed energy equals the total
attributable GPU energy (idle-subtracted) over the same time window.

Error < 5% → PASS.

Requires: vLLM running on port 8000 (NOT the proxy).
"""

import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attribution_engine import AttributionEngine, PowerBuffer
from collectors.gpu_power import GpuPowerSamplerNVML

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("validate3")

VLLM_URL = "http://localhost:8000/v1/chat/completions"


def _detect_model() -> str:
    try:
        resp = httpx.get("http://localhost:8000/v1/models", timeout=10.0)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "Qwen/Qwen2.5-3B-Instruct"


MODEL = _detect_model()
RESULTS_DIR = Path(__file__).resolve().parent / "results"

REQUESTS = [
    {"label": "R1(200)", "prompt": "Explain what a neural network is.", "max_tokens": 200},
    {"label": "R2(400)", "prompt": "Explain how gradient descent works.", "max_tokens": 400},
    {"label": "R3(600)", "prompt": "Explain the transformer architecture in detail.", "max_tokens": 600},
]


def send_request(
    buffer: PowerBuffer,
    engine: AttributionEngine,
    prompt: str,
    max_tokens: int,
    request_id: str,
    results_out: dict,
    label: str,
) -> None:
    """Send a single request to vLLM, recording timing and token counts."""
    client = httpx.Client(timeout=120.0)
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start_time = time.time()
    buffer.register_active_request(request_id, start_time)
    engine.register_request(request_id, start_time)

    try:
        response = client.post(VLLM_URL, json=request_body)
        response.raise_for_status()
        response_json = response.json()
    except Exception as e:
        buffer.unregister_active_request(request_id)
        logger.error("%s failed: %s", label, e)
        results_out["error"] = str(e)
        return
    finally:
        client.close()

    end_time = time.time()
    buffer.unregister_active_request(request_id)

    usage = response_json.get("usage", {})
    results_out.update({
        "request_id": request_id,
        "start_time": start_time,
        "end_time": end_time,
        "latency": end_time - start_time,
        "prefill_tokens": usage.get("prompt_tokens", 0),
        "decode_tokens": usage.get("completion_tokens", 0),
    })
    logger.info(
        "%s done: prefill=%d, decode=%d, latency=%.3fs",
        label,
        results_out["prefill_tokens"],
        results_out["decode_tokens"],
        results_out["latency"],
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Start sampler + buffer + engine ---
    sampler = GpuPowerSamplerNVML(interval_ms=10)
    sampler.start()
    logger.info("Sampler started (10ms interval)")

    time.sleep(1.0)

    buffer = PowerBuffer()
    engine = AttributionEngine(power_buffer=buffer, db_conn=None)

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

    # --- Step 2: Send three requests concurrently ---
    results = [{} for _ in REQUESTS]
    request_ids = [str(uuid.uuid4()) for _ in REQUESTS]

    global_start = time.time()

    threads = []
    for i, req in enumerate(REQUESTS):
        t = threading.Thread(
            target=send_request,
            args=(buffer, engine, req["prompt"], req["max_tokens"],
                  request_ids[i], results[i], req["label"]),
        )
        threads.append(t)

    logger.info("Launching %d requests simultaneously...", len(REQUESTS))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check for errors
    for i, r in enumerate(results):
        if "error" in r:
            logger.error("Request %s failed, aborting.", REQUESTS[i]["label"])
            sampler.stop()
            return

    # Wait for trailing samples
    time.sleep(0.3)

    # --- Step 3: Finalize all requests before attribution ---
    logger.info("Finalizing all requests in registry...")
    for i, r in enumerate(results):
        engine.finalize_request(
            r["request_id"], r["end_time"],
            r["prefill_tokens"], r["decode_tokens"],
        )

    # --- Step 4: Run attribution for each ---
    logger.info("Running attribution...")
    attr_results = []
    for i, r in enumerate(results):
        attr = engine.attribute(
            request_id=r["request_id"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            prefill_tokens=r["prefill_tokens"],
            decode_tokens=r["decode_tokens"],
            endpoint="/v1/chat/completions",
            model=MODEL,
        )
        attr_results.append(attr)

    global_end = max(r["end_time"] for r in results)

    # --- Step 5: Energy conservation check ---
    logger.info("Running energy conservation check...")
    conservation = engine.validate_energy_conservation(
        request_ids, global_start, global_end,
    )

    # Stop feeder
    feed_stop.set()
    feeder.join(timeout=2)

    # --- Step 6: Print results ---
    print()
    print("=" * 60)
    print("VALIDATION 3: Batch-Level Energy Conservation")
    print("=" * 60)
    print()

    total_energy = sum(a["energy_joules"] for a in attr_results)
    for i, req in enumerate(REQUESTS):
        r = results[i]
        a = attr_results[i]
        ratio = a["energy_joules"] / total_energy * 100 if total_energy > 0 else 0
        print(f"=== {req['label']} ===")
        print(f"  Tokens:    {r['prefill_tokens']} prefill, {r['decode_tokens']} decode")
        print(f"  Latency:   {r['latency']:.3f}s")
        print(f"  Energy:    {a['energy_joules']:.4f} J  ({ratio:.1f}%)")
        print()

    print(f"=== Conservation Result ===")
    print(f"  Total attributed:    {conservation['total_attributed_joules']:.4f} J")
    print(f"  Total attributable:  {conservation['total_attributable_joules']:.4f} J")
    print(f"  Error:               {conservation['error_pct']:.2f}%")
    print(f"  Samples used:        {conservation['sample_count']}")
    print(f"  Result:              {'PASS' if conservation['passed'] else 'FAIL'}")
    print("=" * 60)

    # --- Step 7: Save result ---
    output = {
        "requests": [],
        "conservation": conservation,
    }
    for i, req in enumerate(REQUESTS):
        r = results[i]
        a = attr_results[i]
        output["requests"].append({
            "label": req["label"],
            "prefill_tokens": r["prefill_tokens"],
            "decode_tokens": r["decode_tokens"],
            "latency_s": round(r["latency"], 4),
            "energy_joules": a["energy_joules"],
        })

    result_path = RESULTS_DIR / "result.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResult saved to {result_path}")

    # Cleanup
    sampler.stop()


if __name__ == "__main__":
    main()
