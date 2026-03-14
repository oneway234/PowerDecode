"""PowerDecode Cluster Calibration — measure IDLE_POWER, W_PREFILL, W_DECODE.

Run AFTER diagnose.py, BEFORE validate_all.py.
Requires: vLLM running on port 8000.

Outputs calibrate_result.json with the three constants, plus a
copy-paste snippet for attribution_engine.py.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_engine import PowerBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("calibrate")

VLLM_URL = "http://localhost:8000/v1/chat/completions"
RESULT_PATH = Path(__file__).resolve().parent / "calibrate_result.json"
ROUNDS = 5


def _detect_model() -> str:
    """Auto-detect model ID from vLLM."""
    r = httpx.get("http://localhost:8000/v1/models", timeout=10.0)
    r.raise_for_status()
    models = r.json().get("data", [])
    if not models:
        raise RuntimeError("vLLM has no models loaded")
    return models[0]["id"]


def _create_sampler():
    """Create the right sampler based on GPU count."""
    diagnose_path = Path(__file__).resolve().parent / "diagnose_result.json"
    gpu_count = 1
    interval_ms = 10
    if diagnose_path.exists():
        diag = json.loads(diagnose_path.read_text())
        gpu_count = diag.get("gpu_count", 1)
        interval_ms = diag.get("recommended_sampler_interval_ms", 10)

    if gpu_count > 1:
        from collectors.gpu_power import MultiGpuPowerSamplerNVML
        logger.info("Multi-GPU detected (%d GPUs), using MultiGpuPowerSamplerNVML", gpu_count)
        return MultiGpuPowerSamplerNVML(interval_ms=interval_ms), gpu_count
    else:
        from collectors.gpu_power import GpuPowerSamplerNVML
        return GpuPowerSamplerNVML(interval_ms=interval_ms), gpu_count


def _start_feeder(sampler, buffer):
    """Start background thread feeding sampler → buffer."""
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


def _measure_attributable_energy(buffer, start_time, end_time, idle_power):
    """Trapezoidal integration of (power - idle) over a time window."""
    samples = buffer.query(start_time, end_time + 0.05)
    energy = 0.0
    for i in range(len(samples) - 1):
        t1, w1 = samples[i]
        t2, w2 = samples[i + 1]
        interval = t2 - t1
        energy += max(0.0, (w1 + w2) / 2 - idle_power) * interval
    return energy, len(samples)


def _get_prefill_prompt(model_name: str) -> str:
    """Dynamic prefill prompt based on model size.

    Smaller models prefill faster — need longer prompts to ensure
    enough power samples (target: prefill >= 300ms = 30 samples at 10ms).
    """
    base = (
        "The following is a detailed analysis of GPU computing architecture, "
        "memory bandwidth, inference optimization, and power consumption patterns. "
    ) * 3  # ~36 words per block

    model_lower = model_name.lower()
    if "0.5b" in model_lower:
        return base * 8  # ~864 words ≈ 1000 tokens
    elif "1.5b" in model_lower or "1b" in model_lower:
        return base * 5  # ~540 words ≈ 600 tokens
    elif "3b" in model_lower:
        return base * 10  # ~1080 words ≈ 1200 tokens
    elif "7b" in model_lower or "8b" in model_lower:
        return base * 3  # ~324 words ≈ 400 tokens
    else:
        # 13B+ or unknown — short prompt is fine
        return base * 2  # ~216 words ≈ 250 tokens


def _get_decode_max_tokens(model_name: str) -> int:
    """Dynamic decode max_tokens based on model size.

    Smaller models generate faster — need more tokens to accumulate
    enough energy for accurate measurement.
    """
    model_lower = model_name.lower()
    if "0.5b" in model_lower or "1.5b" in model_lower or "1b" in model_lower:
        return 600
    elif "3b" in model_lower:
        return 400
    else:
        return 400


def main() -> None:
    try:
        model_id = _detect_model()
    except Exception:
        print("\nERROR: vLLM not running on port 8000.")
        print("Start vLLM first, then re-run this script.")
        sys.exit(1)
    logger.info("Detected model: %s", model_id)

    sampler, gpu_count = _create_sampler()
    sampler.start()
    logger.info("Sampler started")
    time.sleep(1.0)

    buffer = PowerBuffer()
    feed_stop, feeder = _start_feeder(sampler, buffer)

    # ==================================================================
    # Phase 1: Idle Power (10 seconds, no load)
    # ==================================================================
    logger.info("Phase 1: Measuring idle power (10s)...")
    time.sleep(2.0)  # let GPU settle from any prior activity

    idle_start = time.time()
    time.sleep(10.0)
    idle_end = time.time()

    idle_samples = buffer.query(idle_start, idle_end)
    if len(idle_samples) < 2:
        logger.error("Not enough idle samples, aborting")
        sampler.stop()
        return

    idle_power = sum(w for _, w in idle_samples) / len(idle_samples)
    logger.info("  Idle power: %.2f W (%d samples)", idle_power, len(idle_samples))

    # ==================================================================
    # Phase 2: Prefill Weight (5 rounds, prefill-heavy)
    # ==================================================================
    logger.info("Phase 2: Measuring prefill weight (%d rounds)...", ROUNDS)
    prefill_prompt = _get_prefill_prompt(model_id)
    logger.info("  Prefill prompt length: ~%d words", len(prefill_prompt.split()))
    prefill_energies = []
    prefill_token_counts = []

    client = httpx.Client(timeout=120.0)

    for r in range(ROUNDS):
        time.sleep(2.0)  # settle

        start_time = time.time()
        resp = client.post(VLLM_URL, json={
            "model": model_id,
            "messages": [{"role": "user", "content": prefill_prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
        })
        resp.raise_for_status()
        end_time = time.time()

        usage = resp.json().get("usage", {})
        prefill_tokens = usage.get("prompt_tokens", 0)
        time.sleep(0.3)

        energy, n_samples = _measure_attributable_energy(buffer, start_time, end_time, idle_power)
        prefill_energies.append(energy)
        prefill_token_counts.append(prefill_tokens)
        logger.info("  Round %d: %.4f J, %d tokens, %d samples",
                     r + 1, energy, prefill_tokens, n_samples)

    # Energy per prefill token
    w_prefill_values = [
        e / t for e, t in zip(prefill_energies, prefill_token_counts) if t > 0
    ]
    w_prefill = sum(w_prefill_values) / len(w_prefill_values) if w_prefill_values else 0.0

    # Stability check: warn if prefill rounds vary > 30%
    if w_prefill > 0 and len(w_prefill_values) >= 2:
        max_v = max(w_prefill_values)
        min_v = min(w_prefill_values)
        spread = (max_v - min_v) / w_prefill * 100
        if spread > 30:
            logger.warning(
                "  Prefill W unstable (spread %.0f%%). "
                "Consider longer prompt or more rounds.", spread,
            )

    # ==================================================================
    # Phase 3: Decode Weight (5 rounds, decode-heavy)
    # ==================================================================
    logger.info("Phase 3: Measuring decode weight (%d rounds)...", ROUNDS)
    decode_max_tokens = _get_decode_max_tokens(model_id)
    logger.info("  Decode max_tokens: %d", decode_max_tokens)
    decode_energies = []
    decode_token_counts = []
    decode_prefill_counts = []

    for r in range(ROUNDS):
        time.sleep(2.0)

        start_time = time.time()
        decode_prompt = (
            "Write a detailed step-by-step tutorial on how to build a REST API "
            "with Python Flask. Include code examples for each step, explain "
            "authentication, database integration, error handling, and deployment. "
            "Be thorough and comprehensive."
        )
        resp = client.post(VLLM_URL, json={
            "model": model_id,
            "messages": [{"role": "user", "content": decode_prompt}],
            "max_tokens": decode_max_tokens,
            "temperature": 0.0,
        })
        resp.raise_for_status()
        end_time = time.time()

        usage = resp.json().get("usage", {})
        prefill_tokens = usage.get("prompt_tokens", 0)
        decode_tokens = usage.get("completion_tokens", 0)
        time.sleep(0.3)

        energy, n_samples = _measure_attributable_energy(buffer, start_time, end_time, idle_power)
        decode_energies.append(energy)
        decode_token_counts.append(decode_tokens)
        decode_prefill_counts.append(prefill_tokens)
        logger.info("  Round %d: %.4f J, %d prefill + %d decode tokens, %d samples",
                     r + 1, energy, prefill_tokens, decode_tokens, n_samples)

    client.close()

    # Subtract prefill contribution, then divide by decode tokens
    w_decode_values = []
    for e, pt, dt in zip(decode_energies, decode_prefill_counts, decode_token_counts):
        if dt > 0:
            prefill_contribution = pt * w_prefill
            decode_energy = max(0.0, e - prefill_contribution)
            w_decode_values.append(decode_energy / dt)
    w_decode = sum(w_decode_values) / len(w_decode_values) if w_decode_values else 0.0

    # Cleanup
    feed_stop.set()
    feeder.join(timeout=2)
    sampler.stop()

    # ==================================================================
    # Results
    # ==================================================================
    ratio = w_decode / w_prefill if w_prefill > 0 else 0.0

    result = {
        "gpu_model": "auto",
        "gpu_count": gpu_count,
        "idle_power": round(idle_power, 2),
        "w_prefill": round(w_prefill, 4),
        "w_decode": round(w_decode, 4),
        "decode_prefill_ratio": round(ratio, 1),
        "detail": {
            "idle_samples": len(idle_samples),
            "prefill_rounds": [
                {"energy_j": round(e, 4), "tokens": t}
                for e, t in zip(prefill_energies, prefill_token_counts)
            ],
            "decode_rounds": [
                {"energy_j": round(e, 4), "prefill_tokens": pt, "decode_tokens": dt}
                for e, pt, dt in zip(decode_energies, decode_prefill_counts, decode_token_counts)
            ],
        },
    }

    # Try to fill gpu_model from diagnose_result
    diagnose_path = Path(__file__).resolve().parent / "diagnose_result.json"
    if diagnose_path.exists():
        diag = json.loads(diagnose_path.read_text())
        result["gpu_model"] = diag.get("gpu_model", "unknown")

    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print()
    print("=" * 55)
    print("PowerDecode Calibration Result")
    print("=" * 55)
    print(f"GPU:            {result['gpu_model']} x{gpu_count}")
    print(f"IDLE_POWER:     {idle_power:.2f} W")
    print(f"W_PREFILL:      {w_prefill:.4f} J/token")
    print(f"W_DECODE:       {w_decode:.4f} J/token")
    print(f"Ratio:          {ratio:.1f}x  (decode / prefill)")
    print()
    print(">>> Copy-paste into attribution_engine.py:")
    print()
    print(f"    IDLE_POWER = {idle_power:.2f}")
    print(f"    W_PREFILL  = {w_prefill:.4f}")
    print(f"    W_DECODE   = {w_decode:.4f}")
    print()
    print("=" * 55)
    print(f"Result saved to {RESULT_PATH}")

    # Auto-update baseline database
    try:
        import subprocess
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0].strip()

        from baseline_db import upsert_baseline
        upsert_baseline(
            model=model_id,
            gpu=gpu_name,
            idle_power_w=idle_power,
            w_prefill=w_prefill,
            w_decode=w_decode,
        )
        print(f"Baseline updated in benchmark_baselines.json ({model_id} on {gpu_name})")
    except Exception as e:
        print(f"Warning: could not update baseline DB: {e}")


if __name__ == "__main__":
    main()
