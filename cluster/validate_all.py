"""PowerDecode Cluster Validation — run all 5 validations in sequence.

Run AFTER calibrate.py (constants must be updated in attribution_engine.py).
Requires: vLLM running on port 8000.

Auto-detects model from vLLM. Uses constants from attribution_engine.py.
Saves validate_result.json with pass/fail for each validation.

Validations:
  1. Single request closed-loop (error < 10%)
  2. Dual request concurrent attribution (ratio error < 15%)
  3. Batch-level energy conservation (error < 5%)
  4. Power linearity assumption (coefficient in [0.85, 1.15])
  5. Prefill/Decode ratio stability (CV < 30%)
"""

import json
import logging
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_engine import AttributionEngine, PowerBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("validate_all")

VLLM_URL = "http://localhost:8000/v1/chat/completions"
RESULT_PATH = Path(__file__).resolve().parent / "validate_result.json"


def _detect_model() -> str:
    r = httpx.get("http://localhost:8000/v1/models", timeout=10.0)
    r.raise_for_status()
    models = r.json().get("data", [])
    if not models:
        raise RuntimeError("vLLM has no models loaded")
    return models[0]["id"]


def _detect_gpu() -> dict:
    """Return GPU info dict from diagnose_result.json or defaults."""
    diagnose_path = Path(__file__).resolve().parent / "diagnose_result.json"
    if diagnose_path.exists():
        diag = json.loads(diagnose_path.read_text())
        return {
            "gpu_model": diag.get("gpu_model", "unknown"),
            "gpu_count": diag.get("gpu_count", 1),
            "gpu_memory_gb": diag.get("gpu_memory_gb", 0),
            "sampling_ms": diag.get("recommended_sampler_interval_ms", 10),
        }
    # Fallback: try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        count = pynvml.nvmlDeviceGetCount()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "gpu_model": name,
            "gpu_count": count,
            "gpu_memory_gb": round(mem.total / (1024**3), 1),
            "sampling_ms": 10,
        }
    except Exception:
        return {"gpu_model": "unknown", "gpu_count": 1, "gpu_memory_gb": 0, "sampling_ms": 10}


def _create_sampler(gpu_info: dict):
    gpu_count = gpu_info.get("gpu_count", 1)
    interval_ms = gpu_info.get("sampling_ms", 10)

    if gpu_count > 1:
        from collectors.gpu_power import MultiGpuPowerSamplerNVML
        return MultiGpuPowerSamplerNVML(interval_ms=interval_ms)
    else:
        from collectors.gpu_power import GpuPowerSamplerNVML
        return GpuPowerSamplerNVML(interval_ms=interval_ms)


def _start_feeder(sampler, buffer):
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
    return feed_stop, feeder


def _send_request(model, prompt, max_tokens):
    """Send a blocking request to vLLM. Returns (start, end, prefill, decode)."""
    client = httpx.Client(timeout=120.0)
    start_time = time.time()
    resp = client.post(VLLM_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    resp.raise_for_status()
    end_time = time.time()
    usage = resp.json().get("usage", {})
    client.close()
    return (
        start_time, end_time,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


def _send_request_threaded(model, prompt, max_tokens, buffer, engine, result_out):
    """Thread target: send request with buffer/engine registration."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    buffer.register_active_request(request_id, start_time)
    engine.register_request(request_id, start_time)

    client = httpx.Client(timeout=120.0)
    try:
        resp = client.post(VLLM_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        })
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
    except Exception as e:
        buffer.unregister_active_request(request_id)
        result_out["error"] = str(e)
        return
    finally:
        client.close()

    end_time = time.time()
    buffer.unregister_active_request(request_id)

    result_out.update({
        "request_id": request_id,
        "start_time": start_time,
        "end_time": end_time,
        "prefill_tokens": usage.get("prompt_tokens", 0),
        "decode_tokens": usage.get("completion_tokens", 0),
    })


# ======================================================================
# Validation 1: Single request closed-loop
# ======================================================================

def run_validation1(model, sampler, buffer, engine):
    """Direct measurement vs attribution engine — error < 10%."""
    logger.info("=== Validation 1: Single Request Closed-Loop ===")

    idle_power = engine.IDLE_POWER
    time.sleep(2.0)

    # Send request
    prompt = "hello " * 100
    request_id = str(uuid.uuid4())
    start_time = time.time()
    buffer.register_active_request(request_id, start_time)
    engine.register_request(request_id, start_time)

    start, end, prefill, decode = _send_request(model, prompt, 300)
    buffer.unregister_active_request(request_id)
    time.sleep(0.2)

    # Direct measurement from raw sampler
    raw_samples = [
        (s.timestamp.timestamp(), s.power_watts)
        for s in sampler.samples
        if start <= s.timestamp.timestamp() <= end + 0.05
    ]
    direct_energy = 0.0
    if len(raw_samples) >= 2:
        for i in range(len(raw_samples) - 1):
            t1, w1 = raw_samples[i]
            t2, w2 = raw_samples[i + 1]
            direct_energy += max(0.0, (w1 + w2) / 2 - idle_power) * (t2 - t1)

    # Attribution engine
    engine.finalize_request(request_id, end, prefill, decode)
    attr = engine.attribute(
        request_id=request_id, start_time=start, end_time=end,
        prefill_tokens=prefill, decode_tokens=decode,
        endpoint="/v1/chat/completions", model=model,
    )
    attributed_energy = attr["energy_joules"]

    error_pct = abs(attributed_energy - direct_energy) / direct_energy * 100 if direct_energy > 0 else 100.0
    passed = error_pct < 10

    logger.info("  Direct: %.4f J, Attributed: %.4f J, Error: %.2f%% → %s",
                direct_energy, attributed_energy, error_pct, "PASS" if passed else "FAIL")

    return {
        "name": "Single request closed-loop",
        "threshold": "error < 10%",
        "direct_joules": round(direct_energy, 6),
        "attributed_joules": attributed_energy,
        "error_pct": round(error_pct, 2),
        "passed": passed,
    }


# ======================================================================
# Validation 2: Dual request concurrent attribution
# ======================================================================

def run_validation2(model, sampler, buffer, engine):
    """Two concurrent requests — ratio error < 15%."""
    logger.info("=== Validation 2: Dual Request Concurrent ===")

    time.sleep(2.0)

    result_x, result_y = {}, {}
    tx = threading.Thread(
        target=_send_request_threaded,
        args=(model, "Write a concise overview of machine learning.",
              200, buffer, engine, result_x),
    )
    ty = threading.Thread(
        target=_send_request_threaded,
        args=(model,
              "Write a detailed step-by-step tutorial on how to build a REST API with Python Flask. Include code examples for each step, explain authentication, database integration, error handling, and deployment. Be thorough and comprehensive.",
              600, buffer, engine, result_y),
    )

    tx.start(); ty.start()
    tx.join(); ty.join()

    if "error" in result_x or "error" in result_y:
        logger.error("  Request failed, skipping validation 2")
        return {"name": "Concurrent ratio", "passed": False, "error": "request failed"}

    time.sleep(0.3)

    # Finalize both before attribution
    engine.finalize_request(result_x["request_id"], result_x["end_time"],
                            result_x["prefill_tokens"], result_x["decode_tokens"])
    engine.finalize_request(result_y["request_id"], result_y["end_time"],
                            result_y["prefill_tokens"], result_y["decode_tokens"])

    attr_x = engine.attribute(
        request_id=result_x["request_id"], start_time=result_x["start_time"],
        end_time=result_x["end_time"], prefill_tokens=result_x["prefill_tokens"],
        decode_tokens=result_x["decode_tokens"],
        endpoint="/v1/chat/completions", model=model,
    )
    attr_y = engine.attribute(
        request_id=result_y["request_id"], start_time=result_y["start_time"],
        end_time=result_y["end_time"], prefill_tokens=result_y["prefill_tokens"],
        decode_tokens=result_y["decode_tokens"],
        endpoint="/v1/chat/completions", model=model,
    )

    # Theory: mirror _compute_energy logic
    w_p, w_d = engine.W_PREFILL, engine.W_DECODE
    x_weighted = result_x["prefill_tokens"] * w_p + result_x["decode_tokens"] * w_d
    y_weighted = result_y["prefill_tokens"] * w_p + result_y["decode_tokens"] * w_d

    sx, ex = result_x["start_time"], result_x["end_time"]
    sy, ey = result_y["start_time"], result_y["end_time"]
    full_start, full_end = min(sx, sy), max(ex, ey)
    all_samples = buffer.query(full_start, full_end + 0.05)

    theory_x, theory_y = 0.0, 0.0
    for i in range(len(all_samples) - 1):
        t1, w1 = all_samples[i]
        t2, w2 = all_samples[i + 1]
        interval = t2 - t1
        attr_w = max(0.0, (w1 + w2) / 2 - engine.IDLE_POWER)
        mid = (t1 + t2) / 2
        xa = sx <= mid <= ex
        ya = sy <= mid <= ey
        if xa and ya:
            tw = x_weighted + y_weighted
            theory_x += attr_w * interval * (x_weighted / tw)
            theory_y += attr_w * interval * (y_weighted / tw)
        elif xa:
            theory_x += attr_w * interval
        elif ya:
            theory_y += attr_w * interval

    total_theory = theory_x + theory_y
    total_actual = attr_x["energy_joules"] + attr_y["energy_joules"]

    if total_theory > 0 and total_actual > 0:
        th_x_ratio = theory_x / total_theory
        ac_x_ratio = attr_x["energy_joules"] / total_actual
        th_y_ratio = theory_y / total_theory
        ac_y_ratio = attr_y["energy_joules"] / total_actual
    else:
        th_x_ratio = ac_x_ratio = th_y_ratio = ac_y_ratio = 0.0

    x_err = abs(ac_x_ratio - th_x_ratio) / th_x_ratio * 100 if th_x_ratio > 0 else 0
    y_err = abs(ac_y_ratio - th_y_ratio) / th_y_ratio * 100 if th_y_ratio > 0 else 0
    passed = x_err < 15 and y_err < 15

    logger.info("  X: theory=%.4f actual=%.4f err=%.2f%%", th_x_ratio, ac_x_ratio, x_err)
    logger.info("  Y: theory=%.4f actual=%.4f err=%.2f%%", th_y_ratio, ac_y_ratio, y_err)
    logger.info("  → %s", "PASS" if passed else "FAIL")

    return {
        "name": "Concurrent ratio",
        "threshold": "ratio error < 15%",
        "x_theory_ratio": round(th_x_ratio, 4),
        "x_actual_ratio": round(ac_x_ratio, 4),
        "x_error_pct": round(x_err, 2),
        "y_theory_ratio": round(th_y_ratio, 4),
        "y_actual_ratio": round(ac_y_ratio, 4),
        "y_error_pct": round(y_err, 2),
        "passed": passed,
    }


# ======================================================================
# Validation 3: Batch-level energy conservation
# ======================================================================

def run_validation3(model, sampler, buffer, engine):
    """Three concurrent requests — conservation error < 5%."""
    logger.info("=== Validation 3: Batch Energy Conservation ===")

    time.sleep(2.0)

    prompts = [
        ("Explain what a neural network is.", 200),
        ("Explain how gradient descent works.", 400),
        ("Explain the transformer architecture in detail.", 600),
    ]

    results = [{} for _ in prompts]
    threads = []
    for i, (prompt, max_tok) in enumerate(prompts):
        t = threading.Thread(
            target=_send_request_threaded,
            args=(model, prompt, max_tok, buffer, engine, results[i]),
        )
        threads.append(t)

    global_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in results:
        if "error" in r:
            logger.error("  Request failed, skipping validation 3")
            return {"name": "Energy conservation", "passed": False, "error": "request failed"}

    time.sleep(0.3)

    # Finalize all before attribution
    for r in results:
        engine.finalize_request(r["request_id"], r["end_time"],
                                r["prefill_tokens"], r["decode_tokens"])

    request_ids = []
    for r in results:
        attr = engine.attribute(
            request_id=r["request_id"], start_time=r["start_time"],
            end_time=r["end_time"], prefill_tokens=r["prefill_tokens"],
            decode_tokens=r["decode_tokens"],
            endpoint="/v1/chat/completions", model=model,
        )
        request_ids.append(r["request_id"])

    global_end = max(r["end_time"] for r in results)

    conservation = engine.validate_energy_conservation(
        request_ids, global_start, global_end,
    )
    conservation["name"] = "Energy conservation"
    conservation["threshold"] = "error < 5%"

    logger.info("  Attributed: %.4f J, Attributable: %.4f J, Error: %.2f%% → %s",
                conservation["total_attributed_joules"],
                conservation["total_attributable_joules"],
                conservation["error_pct"],
                "PASS" if conservation["passed"] else "FAIL")

    return conservation


# ======================================================================
# Validation 4: Power linearity assumption
# ======================================================================

def run_validation4(model, sampler, buffer, engine):
    """N concurrent identical requests — power linearity check.

    Runs concurrency 1 and 2 for decode-heavy workload, 2 rounds each.
    Linearity coefficient in [0.85, 1.15] → PASS.

    Note: This validation is expected to FAIL — GPU power is NOT linear
    with concurrency due to vLLM continuous batching. The result documents
    this behavior rather than gating on it.
    """
    logger.info("=== Validation 4: Power Linearity ===")

    idle_power = engine.IDLE_POWER
    rounds = 2
    prompt = "Write a detailed explanation of how neural networks learn."
    max_tokens = 300

    def measure_avg_watts(concurrency: int) -> list[float]:
        round_watts = []
        for r in range(rounds):
            time.sleep(2.0)
            window_start = time.time()

            if concurrency == 1:
                _send_request(model, prompt, max_tokens)
            else:
                results_out = [{} for _ in range(concurrency)]
                threads = []
                for i in range(concurrency):
                    t = threading.Thread(
                        target=_send_request_threaded,
                        args=(model, prompt, max_tokens, buffer, engine, results_out[i]),
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            window_end = time.time()
            time.sleep(0.3)

            samples = buffer.query(window_start, window_end + 0.05)
            duration = window_end - window_start
            if len(samples) < 2 or duration <= 0:
                round_watts.append(0.0)
                continue

            total_energy = 0.0
            for i in range(len(samples) - 1):
                t1, w1 = samples[i]
                t2, w2 = samples[i + 1]
                total_energy += max(0.0, (w1 + w2) / 2 - idle_power) * (t2 - t1)
            round_watts.append(total_energy / duration)
        return round_watts

    # Run measurements
    c1_watts = measure_avg_watts(1)
    c2_watts = measure_avg_watts(2)

    baseline = sum(c1_watts) / len(c1_watts) if c1_watts else 0
    c2_avg = sum(c2_watts) / len(c2_watts) if c2_watts else 0
    linearity = c2_avg / (2 * baseline) if baseline > 0 else 0
    passed = 0.85 <= linearity <= 1.15

    logger.info("  C1 baseline: %.2f W  (rounds: %s)", baseline,
                ", ".join(f"{w:.2f}" for w in c1_watts))
    logger.info("  C2 avg: %.2f W  linearity: %.3f → %s", c2_avg, linearity,
                "PASS" if passed else "FAIL (expected)")

    return {
        "name": "Power linearity",
        "threshold": "coefficient in [0.85, 1.15]",
        "note": "Expected to FAIL — GPU power is constant under vLLM continuous batching",
        "baseline_watts": round(baseline, 2),
        "c2_avg_watts": round(c2_avg, 2),
        "c1_rounds": [round(w, 2) for w in c1_watts],
        "c2_rounds": [round(w, 2) for w in c2_watts],
        "linearity": round(linearity, 3),
        "passed": passed,
    }


# ======================================================================
# Validation 5: Prefill/Decode ratio stability
# ======================================================================

def run_validation5(model, sampler, buffer, engine):
    """Measure prefill and decode power independently, check ratio stability.

    Sends 3 rounds of prefill-heavy then decode-heavy requests.
    Checks that the W_DECODE/W_PREFILL ratio is > 1.0 and CV < 30%.
    """
    logger.info("=== Validation 5: Prefill/Decode Ratio Stability ===")

    idle_power = engine.IDLE_POWER
    rounds = 3
    prefill_prompt = "hello " * 800  # ~1600 tokens
    decode_prompt = "Write a very detailed tutorial on building REST APIs."

    def measure_watts_per_token(prompt, max_tokens, label):
        """Returns list of watts_per_token for each round."""
        results = []
        for r in range(rounds):
            time.sleep(2.0)
            window_start = time.time()
            start, end, prefill, decode = _send_request(model, prompt, max_tokens)
            window_end = end
            time.sleep(0.2)

            samples = buffer.query(window_start, window_end + 0.05)
            if len(samples) < 2:
                results.append(0.0)
                continue

            total_energy = 0.0
            for i in range(len(samples) - 1):
                t1, w1 = samples[i]
                t2, w2 = samples[i + 1]
                total_energy += max(0.0, (w1 + w2) / 2 - idle_power) * (t2 - t1)

            total_tokens = prefill + decode
            wpt = total_energy / total_tokens if total_tokens > 0 else 0
            results.append(wpt)
            logger.info("    %s round %d: %.4f J/token (%d p + %d d = %d tok, %.2f J)",
                        label, r + 1, wpt, prefill, decode, total_tokens, total_energy)
        return results

    prefill_wpt = measure_watts_per_token(prefill_prompt, 1, "prefill")
    decode_wpt = measure_watts_per_token(decode_prompt, 500, "decode")

    avg_prefill = sum(prefill_wpt) / len(prefill_wpt) if prefill_wpt else 0
    avg_decode = sum(decode_wpt) / len(decode_wpt) if decode_wpt else 0
    ratio = avg_decode / avg_prefill if avg_prefill > 0 else 0

    # CV (coefficient of variation) of the ratio across rounds
    if avg_prefill > 0 and len(prefill_wpt) == len(decode_wpt):
        ratios = [d / p if p > 0 else 0 for d, p in zip(decode_wpt, prefill_wpt)]
        mean_ratio = sum(ratios) / len(ratios)
        if mean_ratio > 0 and len(ratios) > 1:
            variance = sum((r - mean_ratio) ** 2 for r in ratios) / (len(ratios) - 1)
            cv = (variance ** 0.5) / mean_ratio * 100
        else:
            cv = 100.0
    else:
        ratios = []
        cv = 100.0

    passed = ratio > 1.0 and cv < 30

    logger.info("  Avg prefill: %.4f J/tok, Avg decode: %.4f J/tok", avg_prefill, avg_decode)
    logger.info("  Ratio (decode/prefill): %.2fx, CV: %.1f%% → %s",
                ratio, cv, "PASS" if passed else "FAIL")

    return {
        "name": "Prefill/Decode ratio stability",
        "threshold": "ratio > 1.0x AND CV < 30%",
        "avg_prefill_j_per_tok": round(avg_prefill, 6),
        "avg_decode_j_per_tok": round(avg_decode, 6),
        "ratio": round(ratio, 2),
        "per_round_ratios": [round(r, 2) for r in ratios],
        "cv_pct": round(cv, 1),
        "passed": passed,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    try:
        model = _detect_model()
    except Exception:
        print("\nERROR: vLLM not running on port 8000.")
        print("Start vLLM first, then re-run this script.")
        sys.exit(1)
    logger.info("Model: %s", model)

    gpu_info = _detect_gpu()
    logger.info("GPU: %s x%d", gpu_info["gpu_model"], gpu_info["gpu_count"])

    sampler = _create_sampler(gpu_info)
    sampler.start()
    logger.info("Sampler started (%dms interval)", gpu_info["sampling_ms"])
    time.sleep(1.0)

    buffer = PowerBuffer()
    engine = AttributionEngine(power_buffer=buffer, db_conn=None)
    feed_stop, feeder = _start_feeder(sampler, buffer)

    # Warm-up
    logger.info("Sending 3 warm-up requests...")
    for i in range(3):
        try:
            _send_request(model, "warmup", 10)
        except Exception:
            pass
        time.sleep(0.5)
    time.sleep(1.0)

    # Run all 5 validations
    v1 = run_validation1(model, sampler, buffer, engine)
    v2 = run_validation2(model, sampler, buffer, engine)
    v3 = run_validation3(model, sampler, buffer, engine)
    v4 = run_validation4(model, sampler, buffer, engine)
    v5 = run_validation5(model, sampler, buffer, engine)

    # Cleanup
    feed_stop.set()
    feeder.join(timeout=2)
    sampler.stop()

    # Summary
    # v4 is informational (expected FAIL), don't gate on it
    core_passed = v1["passed"] and v2["passed"] and v3["passed"]
    all_passed = core_passed and v4["passed"] and v5["passed"]

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "gpu": gpu_info,
            "model": model,
            "constants": {
                "IDLE_POWER": engine.IDLE_POWER,
                "W_PREFILL": engine.W_PREFILL,
                "W_DECODE": engine.W_DECODE,
            },
        },
        "validation1": v1,
        "validation2": v2,
        "validation3": v3,
        "validation4": v4,
        "validation5": v5,
        "core_passed": core_passed,
        "all_passed": all_passed,
        "h100": {
            "_comment": "To be filled when running on Fluidstack H100",
            "meta": None,
            "validation1": None,
            "validation2": None,
            "validation3": None,
            "validation4": None,
            "validation5": None,
            "core_passed": None,
            "all_passed": None,
        },
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n")

    print()
    print("=" * 60)
    print("PowerDecode Cluster Validation Summary")
    print("=" * 60)
    print(f"GPU:          {gpu_info['gpu_model']} x{gpu_info['gpu_count']}")
    print(f"Model:        {model}")
    print(f"IDLE_POWER:   {engine.IDLE_POWER} W")
    print(f"W_PREFILL:    {engine.W_PREFILL}")
    print(f"W_DECODE:     {engine.W_DECODE}")
    print()
    print(f"V1 Single request:        {'PASS' if v1['passed'] else 'FAIL'}  error={v1.get('error_pct', '?')}%")
    print(f"V2 Concurrent ratio:      {'PASS' if v2['passed'] else 'FAIL'}  X={v2.get('x_error_pct', '?')}% Y={v2.get('y_error_pct', '?')}%")
    print(f"V3 Energy conservation:   {'PASS' if v3['passed'] else 'FAIL'}  error={v3.get('error_pct', '?')}%")
    print(f"V4 Power linearity:       {'PASS' if v4['passed'] else 'FAIL'}  coeff={v4.get('linearity', '?')}")
    print(f"V5 PD ratio stability:    {'PASS' if v5['passed'] else 'FAIL'}  ratio={v5.get('ratio', '?')}x CV={v5.get('cv_pct', '?')}%")
    print()
    print(f">>> Core (V1-V3): {'PASS' if core_passed else 'FAIL'}")
    print(f">>> Overall (V1-V5): {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
    print(f"Result saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
