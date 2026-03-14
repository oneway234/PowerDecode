"""Validation 2: Dual-request concurrent attribution verification.

Sends two requests simultaneously (one prefill-heavy, one decode-heavy)
and checks whether the Attribution Engine's energy allocation ratio
matches the theoretical weighted-token ratio.

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
logger = logging.getLogger("validate2")

W_PREFILL = 0.0212
W_DECODE = 0.1772
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

    # --- Step 2: Send two decode-heavy requests concurrently ---
    # Both short prompts, different max_tokens to create known ratio
    prompt_x = "Write a concise overview of machine learning."
    prompt_y = "Write a detailed step-by-step tutorial on how to build a REST API with Python Flask. Include code examples for each step, explain authentication, database integration, error handling, and deployment. Be thorough and comprehensive."

    result_x: dict = {}
    result_y: dict = {}
    request_id_x = str(uuid.uuid4())
    request_id_y = str(uuid.uuid4())

    thread_x = threading.Thread(
        target=send_request,
        args=(buffer, engine, prompt_x, 200, request_id_x, result_x, "X(decode-200)"),
    )
    thread_y = threading.Thread(
        target=send_request,
        args=(buffer, engine, prompt_y, 600, request_id_y, result_y, "Y(decode-600)"),
    )

    logger.info("Launching both requests simultaneously...")
    thread_x.start()
    thread_y.start()
    thread_x.join()
    thread_y.join()

    if "error" in result_x or "error" in result_y:
        logger.error("One or both requests failed, aborting.")
        sampler.stop()
        return

    # Wait for trailing samples
    time.sleep(0.3)

    # --- Step 3: Finalize both requests BEFORE attribution ---
    # Critical: both requests must be finalized in the registry before
    # attribution runs, otherwise the first-attributed request won't see
    # the second as a concurrent request (share=1.0 bug).
    logger.info("Finalizing both requests in registry...")
    engine.finalize_request(
        result_x["request_id"], result_x["end_time"],
        result_x["prefill_tokens"], result_x["decode_tokens"],
    )
    engine.finalize_request(
        result_y["request_id"], result_y["end_time"],
        result_y["prefill_tokens"], result_y["decode_tokens"],
    )

    logger.info("Running attribution...")
    attr_x = engine.attribute(
        request_id=result_x["request_id"],
        start_time=result_x["start_time"],
        end_time=result_x["end_time"],
        prefill_tokens=result_x["prefill_tokens"],
        decode_tokens=result_x["decode_tokens"],
        endpoint="/v1/chat/completions",
        model=MODEL,
    )

    attr_y = engine.attribute(
        request_id=result_y["request_id"],
        start_time=result_y["start_time"],
        end_time=result_y["end_time"],
        prefill_tokens=result_y["prefill_tokens"],
        decode_tokens=result_y["decode_tokens"],
        endpoint="/v1/chat/completions",
        model=MODEL,
    )

    # Stop feeder
    feed_stop.set()
    feeder.join(timeout=2)

    # --- Step 4: Calculate ratios (mirror of _compute_energy logic) ---
    x_weighted = result_x["prefill_tokens"] * W_PREFILL + result_x["decode_tokens"] * W_DECODE
    y_weighted = result_y["prefill_tokens"] * W_PREFILL + result_y["decode_tokens"] * W_DECODE

    start_x, end_x = result_x["start_time"], result_x["end_time"]
    start_y, end_y = result_y["start_time"], result_y["end_time"]

    overlap_start = max(start_x, start_y)
    overlap_end = min(end_x, end_y)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    full_start = min(start_x, start_y)
    full_end = max(end_x, end_y)
    all_samples = buffer.query(full_start, full_end + 0.05)

    # Mirror _compute_energy: iterate through samples, check active requests
    # at each midpoint, split attributable power by weighted token share.
    theory_x_energy = 0.0
    theory_y_energy = 0.0
    total_attr_energy = 0.0
    total_duration = 0.0

    for i in range(len(all_samples) - 1):
        t1, w1 = all_samples[i]
        t2, w2 = all_samples[i + 1]
        interval = t2 - t1
        attributable_watts = max(0.0, (w1 + w2) / 2 - 21.07)
        total_attr_energy += attributable_watts * interval
        total_duration += interval

        midpoint = (t1 + t2) / 2
        x_active = start_x <= midpoint <= end_x
        y_active = start_y <= midpoint <= end_y

        if x_active and y_active:
            total_w = x_weighted + y_weighted
            theory_x_energy += attributable_watts * interval * (x_weighted / total_w)
            theory_y_energy += attributable_watts * interval * (y_weighted / total_w)
        elif x_active:
            theory_x_energy += attributable_watts * interval
        elif y_active:
            theory_y_energy += attributable_watts * interval

    avg_power = total_attr_energy / total_duration if total_duration > 0 else 0.0

    theory_total = theory_x_energy + theory_y_energy
    if theory_total > 0:
        theory_x_ratio = theory_x_energy / theory_total
        theory_y_ratio = theory_y_energy / theory_total
    else:
        theory_x_ratio = 0.5
        theory_y_ratio = 0.5

    # Actual ratios
    total_energy = attr_x["energy_joules"] + attr_y["energy_joules"]
    if total_energy > 0:
        actual_x_ratio = attr_x["energy_joules"] / total_energy
        actual_y_ratio = attr_y["energy_joules"] / total_energy
    else:
        actual_x_ratio = 0.0
        actual_y_ratio = 0.0

    x_error_pct = abs(actual_x_ratio - theory_x_ratio) / theory_x_ratio * 100 if theory_x_ratio > 0 else 0
    y_error_pct = abs(actual_y_ratio - theory_y_ratio) / theory_y_ratio * 100 if theory_y_ratio > 0 else 0

    passed = x_error_pct < 15 and y_error_pct < 15

    # --- Step 5: Print results ---
    print()
    print("=" * 60)
    print("VALIDATION 2: Dual Request Concurrent Attribution")
    print("=" * 60)
    print()
    print(f"=== Time Segments ===")
    print(f"  Overlap:         {overlap_duration:.3f}s")
    print(f"  X duration:      {end_x - start_x:.3f}s")
    print(f"  Y duration:      {end_y - start_y:.3f}s")
    print(f"  Avg power:       {avg_power:.2f}W (attributable)")
    print()
    print(f"=== Request X (decode-200) ===")
    print(f"  Tokens:          {result_x['prefill_tokens']} prefill, {result_x['decode_tokens']} decode")
    print(f"  Latency:         {result_x['latency']:.3f}s")
    print(f"  Weighted tokens: {x_weighted:.4f}")
    print(f"  Energy actual:   {attr_x['energy_joules']:.6f} J")
    print(f"  Energy theory:   {theory_x_energy:.6f} J")
    print(f"  Theory ratio:    {theory_x_ratio:.4f}")
    print(f"  Actual ratio:    {actual_x_ratio:.4f}")
    print()
    print(f"=== Request Y (decode-600) ===")
    print(f"  Tokens:          {result_y['prefill_tokens']} prefill, {result_y['decode_tokens']} decode")
    print(f"  Latency:         {result_y['latency']:.3f}s")
    print(f"  Weighted tokens: {y_weighted:.4f}")
    print(f"  Energy actual:   {attr_y['energy_joules']:.6f} J")
    print(f"  Energy theory:   {theory_y_energy:.6f} J")
    print(f"  Theory ratio:    {theory_y_ratio:.4f}")
    print(f"  Actual ratio:    {actual_y_ratio:.4f}")
    print()
    print(f"=== Validation Result ===")
    print(f"  Total energy:    {total_energy:.6f} J")
    print(f"  X ratio error:   {x_error_pct:.2f}%")
    print(f"  Y ratio error:   {y_error_pct:.2f}%")
    print(f"  Result:          {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    # --- Step 6: Save result ---
    output = {
        "request_x": {
            "prefill_tokens": result_x["prefill_tokens"],
            "decode_tokens": result_x["decode_tokens"],
            "latency_s": round(result_x["latency"], 4),
            "energy_joules": attr_x["energy_joules"],
            "weighted_tokens": round(x_weighted, 4),
            "theory_ratio": round(theory_x_ratio, 4),
            "actual_ratio": round(actual_x_ratio, 4),
            "error_pct": round(x_error_pct, 2),
        },
        "request_y": {
            "prefill_tokens": result_y["prefill_tokens"],
            "decode_tokens": result_y["decode_tokens"],
            "latency_s": round(result_y["latency"], 4),
            "energy_joules": attr_y["energy_joules"],
            "weighted_tokens": round(y_weighted, 4),
            "theory_ratio": round(theory_y_ratio, 4),
            "actual_ratio": round(actual_y_ratio, 4),
            "error_pct": round(y_error_pct, 2),
        },
        "total_energy_joules": round(total_energy, 6),
        "passed": passed,
    }
    result_path = RESULTS_DIR / "result.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResult saved to {result_path}")

    # Cleanup
    sampler.stop()


if __name__ == "__main__":
    main()
