"""PowerDecode Cluster Diagnostic — run first on new hardware.

Detects GPU hardware, sampling capabilities, and recommends a strategy.
Saves result to cluster/diagnose_result.json.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parent / "diagnose_result.json"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    notes: list[str] = []
    gpu_count = 0
    gpu_model = "unknown"
    gpu_memory_gb = 0.0
    nvidia_smi_available = False
    pynvml_available = False
    pynvml_sampling_ms = 0
    dcgm_available = False
    decode_latency_sec = 0.0
    decode_sample_count = 0
    recommended_strategy = "unknown"
    recommended_interval = 100
    warmup_requests_needed = 3

    # ================================================================
    # Step 1: GPU basic info via nvidia-smi
    # ================================================================
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,count",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            nvidia_smi_available = True
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            gpu_count = len(lines)
            if lines:
                parts = lines[0].split(",")
                gpu_model = parts[0].strip()
                gpu_memory_gb = round(float(parts[1].strip()) / 1024, 1) if len(parts) > 1 else 0.0
        else:
            notes.append(f"nvidia-smi returned code {result.returncode}: {result.stderr.strip()}")
    except FileNotFoundError:
        notes.append("nvidia-smi not found in PATH")
    except Exception as e:
        notes.append(f"nvidia-smi error: {e}")

    # ================================================================
    # Step 2: pynvml availability
    # ================================================================
    try:
        import pynvml
        from collectors.gpu_power import _nvml_acquire, _nvml_release
        _nvml_acquire()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetPowerUsage(handle)
        _nvml_release()
        pynvml_available = True
    except Exception as e:
        pynvml_available = False
        notes.append(f"pynvml not available: {e}")

    # ================================================================
    # Step 3: pynvml actual sampling frequency test
    # ================================================================
    if pynvml_available:
        try:
            from collectors.gpu_power import GpuPowerSamplerNVML

            sampler = GpuPowerSamplerNVML(interval_ms=10)
            sampler.start()
            time.sleep(3.0)
            sampler.stop()

            sample_count = len(sampler.samples)
            if sample_count >= 2:
                actual_interval_ms = 3000.0 / sample_count
                if actual_interval_ms <= 12:
                    pynvml_sampling_ms = 10
                elif actual_interval_ms <= 20:
                    pynvml_sampling_ms = 15
                    notes.append(f"10ms unstable, using 15ms (actual avg: {actual_interval_ms:.1f}ms)")
                else:
                    pynvml_sampling_ms = int(actual_interval_ms)
                    notes.append(f"pynvml slower than expected: {actual_interval_ms:.1f}ms avg")
            else:
                pynvml_sampling_ms = 0
                notes.append(f"pynvml sampling test: only {sample_count} samples in 3s")
        except Exception as e:
            notes.append(f"pynvml sampling test error: {e}")

    # ================================================================
    # Step 4: DCGM availability
    # ================================================================
    try:
        result = subprocess.run(
            ["dcgmi", "discovery", "-l"],
            capture_output=True, text=True, timeout=5,
        )
        dcgm_available = result.returncode == 0
    except FileNotFoundError:
        dcgm_available = False
    except Exception:
        dcgm_available = False

    # ================================================================
    # Step 5: Decode latency test (requires vLLM on port 8000)
    # ================================================================
    model_id = None
    try:
        import httpx

        r = httpx.get("http://localhost:8000/v1/models", timeout=3.0)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            model_id = models[0]["id"]
    except Exception:
        notes.append("vLLM not running on port 8000, skipping decode test")

    if model_id:
        try:
            import httpx
            client = httpx.Client(timeout=120.0)
            request_body = {
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 200,
                "temperature": 0.0,
            }

            if pynvml_available:
                from collectors.gpu_power import GpuPowerSamplerNVML
                sampler = GpuPowerSamplerNVML(interval_ms=pynvml_sampling_ms or 10)
                sampler.start()
                time.sleep(0.5)

            start_time = time.time()
            resp = client.post("http://localhost:8000/v1/chat/completions", json=request_body)
            resp.raise_for_status()
            end_time = time.time()
            decode_latency_sec = end_time - start_time

            if pynvml_available:
                time.sleep(0.3)
                sampler.stop()
                # Count samples within the request window
                decode_sample_count = sum(
                    1 for s in sampler.samples
                    if start_time <= s.timestamp.timestamp() <= end_time + 0.05
                )
            else:
                decode_sample_count = int(decode_latency_sec / 0.1)

            client.close()
        except Exception as e:
            notes.append(f"Decode latency test error: {e}")

    # ================================================================
    # Step 6: Recommended strategy
    # ================================================================
    if pynvml_available and pynvml_sampling_ms > 0:
        if pynvml_sampling_ms <= 10:
            recommended_strategy = "pynvml_10ms"
        else:
            recommended_strategy = f"pynvml_{pynvml_sampling_ms}ms"
        recommended_interval = pynvml_sampling_ms
    elif dcgm_available:
        recommended_strategy = "dcgm"
        recommended_interval = 100
        notes.append("DCGM min interval is 100ms, prefill attribution will use energy conservation fallback")
    elif nvidia_smi_available:
        recommended_strategy = "nvidia_smi_fallback"
        recommended_interval = 100
        notes.append("Only nvidia-smi available, using energy conservation fallback for prefill")
    else:
        recommended_strategy = "no_gpu_detected"
        recommended_interval = 0
        notes.append("No GPU sampling method available")

    # ================================================================
    # Step 7: Warm-up estimate
    # ================================================================
    warmup_requests_needed = 3

    # ================================================================
    # Build result
    # ================================================================
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_count": gpu_count,
        "gpu_model": gpu_model,
        "gpu_memory_gb": gpu_memory_gb,
        "pynvml_available": pynvml_available,
        "pynvml_sampling_ms": pynvml_sampling_ms,
        "dcgm_available": dcgm_available,
        "nvidia_smi_available": nvidia_smi_available,
        "decode_latency_sec": round(decode_latency_sec, 3),
        "decode_sample_count": decode_sample_count,
        "recommended_strategy": recommended_strategy,
        "recommended_sampler_interval_ms": recommended_interval,
        "warmup_requests_needed": warmup_requests_needed,
        "notes": notes,
    }

    # Save
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    # Print report
    pynvml_mark = "\u2713" if pynvml_available else "\u2717"
    dcgm_mark = "\u2713" if dcgm_available else "\u2717"
    smi_mark = "\u2713" if nvidia_smi_available else "\u2717"

    print()
    print("=" * 50)
    print("PowerDecode Cluster Diagnostic Report")
    print("=" * 50)
    print(f"GPU:              {gpu_model} x{gpu_count}")
    print(f"VRAM:             {gpu_memory_gb}GB")
    print(f"pynvml:           {pynvml_mark}")
    print(f"DCGM:             {dcgm_mark}")
    print(f"nvidia-smi:       {smi_mark}")
    print(f"Decode latency:   {decode_latency_sec:.3f}s")
    print(f"Decode samples:   {decode_sample_count} points")
    print()
    print(f">>> Recommended strategy: {recommended_strategy}")
    print(f">>> Sampler interval:     {recommended_interval}ms")
    print(f">>> Warm-up requests:     {warmup_requests_needed}")
    print()
    if notes:
        print("Notes:")
        for n in notes:
            print(f"  - {n}")
    print("=" * 50)
    print(f"Result saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
