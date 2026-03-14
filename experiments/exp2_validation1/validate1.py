"""Validation 1: Single-request closed-loop verification.

Verifies that the Attribution Engine's energy calculation matches
an INDEPENDENT direct measurement from raw sampler data.

The two measurements are truly independent:
  - direct_joules: computed from GpuPowerSamplerNVML.samples (raw)
  - attributed_joules: computed by AttributionEngine via PowerBuffer

This validates the full pipeline: sampler → PowerBuffer → AttributionEngine.

Requires: vLLM running on port 8000 (NOT the proxy).
"""

import json
import logging
import sys
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
logger = logging.getLogger("validate1")

IDLE_POWER = 21.07
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


def _feed_sampler_to_buffer(sampler: GpuPowerSamplerNVML, buffer: PowerBuffer) -> None:
    """One-shot: copy all current sampler samples into the PowerBuffer."""
    for s in sampler.samples:
        buffer.append(s.timestamp.timestamp(), s.power_watts)


def _direct_measurement(sampler: GpuPowerSamplerNVML, start_time: float, end_time: float) -> tuple[float, int]:
    """Compute energy directly from raw sampler samples (independent of PowerBuffer).

    Returns (energy_joules, sample_count).
    """
    # Filter raw samples by time window
    raw_samples = []
    for s in sampler.samples:
        ts = s.timestamp.timestamp()
        if start_time <= ts <= end_time + 0.05:
            raw_samples.append((ts, s.power_watts))

    if len(raw_samples) < 2:
        return 0.0, len(raw_samples)

    # Trapezoidal integration
    energy = 0.0
    for i in range(len(raw_samples) - 1):
        t1, w1 = raw_samples[i]
        t2, w2 = raw_samples[i + 1]
        interval = t2 - t1
        attributable_watts = max(0.0, (w1 + w2) / 2 - IDLE_POWER)
        energy += attributable_watts * interval

    return energy, len(raw_samples)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Start power sampler ---
    sampler = GpuPowerSamplerNVML(interval_ms=10)
    sampler.start()
    logger.info("Sampler started (10ms interval)")

    # --- Step 2: Wait for sampler to stabilize ---
    time.sleep(1.0)

    # --- Step 3: Create PowerBuffer and AttributionEngine (no DB) ---
    # PowerBuffer is a SEPARATE data path from the raw sampler.samples
    buffer = PowerBuffer()
    engine = AttributionEngine(power_buffer=buffer, db_conn=None)

    # Start feeding samples into buffer (simulate what proxy._sampler_loop does)
    # We do this in a background thread to mimic real operation
    import threading

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

    # --- Step 4: Send a single request to vLLM ---
    prompt = "hello " * 100  # ~100 tokens
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    request_id = str(uuid.uuid4())
    start_time = time.time()
    buffer.register_active_request(request_id, start_time)

    logger.info("Sending request to vLLM...")
    client = httpx.Client(timeout=120.0)
    response = client.post(VLLM_URL, json=request_body)
    response.raise_for_status()
    response_json = response.json()

    end_time = time.time()
    buffer.unregister_active_request(request_id)

    usage = response_json.get("usage", {})
    prefill_tokens = usage.get("prompt_tokens", 0)
    decode_tokens = usage.get("completion_tokens", 0)
    latency = end_time - start_time

    logger.info(
        "Response: prefill=%d, decode=%d, latency=%.3fs",
        prefill_tokens, decode_tokens, latency,
    )

    # Wait for trailing samples to land in both sampler and buffer
    time.sleep(0.2)

    # --- Step 5: Direct measurement (from RAW sampler, independent of PowerBuffer) ---
    direct_joules, direct_sample_count = _direct_measurement(sampler, start_time, end_time)
    logger.info("Direct measurement: %d raw samples, %.4f J", direct_sample_count, direct_joules)

    # --- Step 6: Attribution Engine result (via PowerBuffer pipeline) ---
    result = engine.attribute(
        request_id=request_id,
        start_time=start_time,
        end_time=end_time,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        endpoint="/v1/chat/completions",
        model=MODEL,
    )
    attributed_joules = result["energy_joules"]

    # Stop feeder
    feed_stop.set()
    feeder.join(timeout=2)

    # --- Step 7: Calculate error ---
    if direct_joules > 0:
        error_pct = abs(attributed_joules - direct_joules) / direct_joules * 100
    else:
        error_pct = 100.0 if attributed_joules > 0 else 0.0

    # --- Step 8: Print results ---
    # Also query buffer to see how many samples it had
    buffer_samples = buffer.query(start_time, end_time + 0.05)

    print()
    print("=" * 60)
    print("VALIDATION 1: Single Request Closed-Loop")
    print("=" * 60)
    print(f"Latency:                   {latency:.3f}s")
    print(f"Prefill tokens:            {prefill_tokens}")
    print(f"Decode tokens:             {decode_tokens}")
    print(f"Raw sampler samples:       {direct_sample_count}")
    print(f"PowerBuffer samples:       {len(buffer_samples)}")
    print(f"Direct measurement:        {direct_joules:.6f} J  (from raw sampler)")
    print(f"Attribution result:        {attributed_joules:.6f} J  (from PowerBuffer)")
    print(f"Error:                     {error_pct:.2f}%")
    passed = error_pct < 10
    print(f"Result:                    {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    if direct_sample_count != len(buffer_samples):
        print(f"\n⚠️  Sample count mismatch: raw={direct_sample_count} vs buffer={len(buffer_samples)}")
        print("   This means PowerBuffer may have lost or duplicated samples.")

    # --- Step 9: Save result ---
    output = {
        "latency_s": round(latency, 4),
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "raw_sample_count": direct_sample_count,
        "buffer_sample_count": len(buffer_samples),
        "direct_joules": round(direct_joules, 6),
        "attributed_joules": attributed_joules,
        "error_pct": round(error_pct, 2),
        "passed": passed,
    }
    result_path = RESULTS_DIR / "result.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResult saved to {result_path}")

    # Cleanup
    sampler.stop()
    client.close()


if __name__ == "__main__":
    main()
