"""PowerDecode Proxy — FastAPI server on port 8001.

Intercepts requests to vLLM (port 8000), measures GPU power consumption,
and computes per-request energy attribution.
"""

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from attribution_engine import AttributionEngine, PowerBuffer
from collectors.gpu_power import GpuPowerSamplerNVML, MultiGpuPowerSamplerNVML
from db import get_recent_requests, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("powerdecode.proxy")

VLLM_BASE = "http://localhost:8000"
WARMUP_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ======================================================================
# Global state — initialized in lifespan
# ======================================================================

_db_conn = None
_power_buffer = None
_engine = None
_sampler = None
_sampler_thread = None


def _sampler_loop(sampler: GpuPowerSamplerNVML, buffer: PowerBuffer) -> None:
    """Bridge: continuously feed sampler samples into the PowerBuffer.

    GpuPowerSamplerNVML API:
      - sampler.samples: list[PowerSample]
      - PowerSample.timestamp: datetime (not float)
      - PowerSample.power_watts: float
      - sampler.is_running: bool property
    """
    last_count = 0
    while sampler.is_running:
        current_count = len(sampler.samples)
        if current_count > last_count:
            for sample in sampler.samples[last_count:current_count]:
                # Convert datetime → float (unix timestamp)
                ts = sample.timestamp.timestamp()
                buffer.append(ts, sample.power_watts)
            last_count = current_count
        time.sleep(0.005)  # 5ms poll interval


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _db_conn, _power_buffer, _engine, _sampler, _sampler_thread

    # --- Startup ---
    _db_conn = init_db()
    _power_buffer = PowerBuffer()
    _engine = AttributionEngine(_power_buffer, _db_conn)

    # Auto-detect multi-GPU from diagnose_result.json
    _diagnose_path = Path(__file__).resolve().parent / "cluster" / "diagnose_result.json"
    _gpu_count = 1
    _interval_ms = 10
    if _diagnose_path.exists():
        import json as _json
        _diag = _json.loads(_diagnose_path.read_text())
        _gpu_count = _diag.get("gpu_count", 1)
        _interval_ms = _diag.get("recommended_sampler_interval_ms", 10)

    if _gpu_count > 1:
        _sampler = MultiGpuPowerSamplerNVML(interval_ms=_interval_ms)
        logger.info("Multi-GPU sampler started (%d GPUs, %dms interval)", _gpu_count, _interval_ms)
    else:
        _sampler = GpuPowerSamplerNVML(interval_ms=_interval_ms)
        logger.info("GPU power sampler started (%dms interval)", _interval_ms)
    _sampler.start()

    _sampler_thread = threading.Thread(
        target=_sampler_loop,
        args=(_sampler, _power_buffer),
        daemon=True,
    )
    _sampler_thread.start()

    logger.info("PowerDecode proxy ready on port 8001")

    # --- Auto warm-up in background (doesn't block startup) ---
    def _run_warmup():
        time.sleep(2.0)  # wait for uvicorn to be fully ready
        logger.info("Starting GPU warm-up (3 requests)...")
        payload = {
            "model": WARMUP_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
        for i in range(3):
            try:
                httpx.post(
                    "http://localhost:8001/v1/chat/completions",
                    json=payload,
                    timeout=60.0,
                    headers={"X-Warmup": "true"},
                )
                logger.info("Warm-up request %d/3 done", i + 1)
            except Exception as e:
                logger.warning("Warm-up request %d/3 failed: %s", i + 1, e)
            time.sleep(0.5)
        logger.info("GPU warm-up complete")

    threading.Thread(target=_run_warmup, daemon=True).start()

    yield

    # --- Shutdown ---
    _sampler.stop()
    if _db_conn:
        _db_conn.close()
    logger.info("PowerDecode proxy shut down")


app = FastAPI(title="PowerDecode Proxy", lifespan=lifespan)

# Reusable httpx client
_http_client = httpx.Client(base_url=VLLM_BASE, timeout=120.0)


# ======================================================================
# Routes
# ======================================================================


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward to vLLM, measure power, compute attribution."""
    is_warmup = request.headers.get("X-Warmup") == "true"
    request_body = await request.json()
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Extract prompt preview from last message
    messages = request_body.get("messages", [])
    last_content = messages[-1].get("content", "") if messages else ""
    prompt_preview = last_content[:60].replace("\n", " ").strip()
    if len(last_content) > 60:
        prompt_preview += "..."
    if is_warmup:
        prompt_preview = "[warm-up]"

    if not is_warmup:
        _power_buffer.register_active_request(request_id, start_time)
        _engine.register_request(request_id, start_time)

    try:
        response = _http_client.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_json = response.json()
    except httpx.HTTPError as e:
        if not is_warmup:
            _power_buffer.unregister_active_request(request_id)
        logger.error("vLLM request failed: %s", e)
        return JSONResponse(
            status_code=502,
            content={"error": f"vLLM upstream error: {e}"},
        )

    end_time = time.time()

    # Warm-up: skip attribution and DB, just return response
    if is_warmup:
        return JSONResponse(content=response_json)

    # --- Normal request: full attribution pipeline ---
    usage = response_json.get("usage", {})
    prefill_tokens = usage.get("prompt_tokens", 0)
    decode_tokens = usage.get("completion_tokens", 0)
    model = request_body.get("model", "unknown")

    _power_buffer.unregister_active_request(request_id)

    # Finalize in registry immediately so concurrent requests see each other
    _engine.finalize_request(request_id, end_time, prefill_tokens, decode_tokens)

    _engine.attribute_async(
        request_id=request_id,
        start_time=start_time,
        end_time=end_time,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        endpoint="/v1/chat/completions",
        model=model,
        prompt_preview=prompt_preview,
    )

    return JSONResponse(content=response_json)


@app.get("/health")
async def health():
    """Health check — also pings vLLM."""
    try:
        r = _http_client.get("/v1/models")
        vllm_status = "ok" if r.status_code == 200 else "error"
    except httpx.HTTPError:
        vllm_status = "error"

    return {"status": "ok", "vllm": vllm_status}


@app.get("/stats/recent")
async def stats_recent(limit: int = 20):
    """Return the most recent N request cost records (default 20, max 500)."""
    limit = min(max(limit, 1), 500)
    rows = get_recent_requests(_db_conn, limit=limit)
    return {"requests": rows, "count": len(rows)}


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=8001, log_level="info")
