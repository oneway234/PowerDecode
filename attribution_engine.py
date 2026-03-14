"""PowerDecode Attribution Engine.

PowerBuffer: thread-safe ring buffer for GPU power samples.
AttributionEngine: per-request energy attribution with weighted token allocation.
"""

import collections
import logging
import sqlite3
import threading
import time
from typing import Optional

from db import insert_request

logger = logging.getLogger(__name__)


# ======================================================================
# PowerBuffer
# ======================================================================


class PowerBuffer:
    """Thread-safe buffer for GPU power samples.

    Stores (timestamp, watts) tuples in a deque. Supports concurrent readers
    (multiple requests querying overlapping time ranges) and a single writer
    (the power sampler thread).

    Old samples are automatically pruned: anything older than the earliest
    active request's start_time (minus 1s buffer). When no requests are
    active, keeps the last 60 seconds.
    """

    def __init__(self) -> None:
        self._buffer: collections.deque[tuple[float, float]] = collections.deque()
        self._active_requests: dict[str, float] = {}  # request_id -> start_time
        self._lock = threading.Lock()

    def append(self, timestamp: float, watts: float) -> None:
        """Add a power sample and prune old data."""
        with self._lock:
            self._buffer.append((timestamp, watts))
            self._cleanup(timestamp)

    def query(self, start_time: float, end_time: float) -> list[tuple[float, float]]:
        """Return all (timestamp, watts) samples where start_time <= t <= end_time."""
        with self._lock:
            return [
                (t, w) for t, w in self._buffer if start_time <= t <= end_time
            ]

    def register_active_request(self, request_id: str, start_time: float) -> None:
        """Mark a request as active so its time range is preserved in the buffer."""
        with self._lock:
            self._active_requests[request_id] = start_time

    def unregister_active_request(self, request_id: str) -> None:
        """Remove a request from the active set."""
        with self._lock:
            self._active_requests.pop(request_id, None)

    @property
    def active_requests(self) -> dict[str, float]:
        """Return a snapshot of currently active requests. {request_id: start_time}"""
        with self._lock:
            return dict(self._active_requests)

    def _cleanup(self, current_timestamp: float) -> None:
        """Remove samples older than the cutoff. Must be called with lock held."""
        if self._active_requests:
            cutoff = min(self._active_requests.values()) - 1.0
        else:
            cutoff = current_timestamp - 60.0

        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()


# ======================================================================
# AttributionEngine
# ======================================================================


class AttributionEngine:
    """Compute per-request energy attribution using weighted token allocation.

    For each request, slices the power buffer by the request's time window,
    subtracts idle power, and distributes the attributable energy proportionally
    based on weighted token counts.
    """

    IDLE_POWER = 21.07            # watts
    W_PREFILL = 0.0212            # W/token
    W_DECODE = 0.1772             # W/token
    WAIT_AFTER_END = 0.05         # seconds — let last samples land
    ENERGY_COST_PER_KWH = 0.12   # USD

    def __init__(
        self,
        power_buffer: PowerBuffer,
        db_conn: sqlite3.Connection,
    ) -> None:
        self._buffer = power_buffer
        self._db_conn = db_conn
        self._registry: dict[str, dict] = {}
        self._registry_lock = threading.Lock()

    def register_request(self, request_id: str, start_time: float) -> None:
        """Register a request at the start of its lifetime."""
        with self._registry_lock:
            self._registry[request_id] = {
                "start_time": start_time,
                "end_time": None,
                "weighted_tokens": None,
            }

    def finalize_request(
        self,
        request_id: str,
        end_time: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        """Fill in end_time and weighted_tokens once the request completes."""
        with self._registry_lock:
            if request_id in self._registry:
                self._registry[request_id]["end_time"] = end_time
                self._registry[request_id]["weighted_tokens"] = (
                    prefill_tokens * self.W_PREFILL
                    + decode_tokens * self.W_DECODE
                )

    def attribute(
        self,
        request_id: str,
        start_time: float,
        end_time: float,
        prefill_tokens: int,
        decode_tokens: int,
        endpoint: str,
        model: str,
        prompt_preview: str = "",
    ) -> dict:
        """Compute energy attribution for a single request (blocking)."""

        # Ensure this request is finalized in the registry
        self.finalize_request(request_id, end_time, prefill_tokens, decode_tokens)

        # Wait a short time for trailing power samples to arrive
        time.sleep(self.WAIT_AFTER_END)

        samples = self._buffer.query(start_time, end_time + self.WAIT_AFTER_END)

        if len(samples) < 2:
            logger.warning(
                "request %s: only %d power sample(s) — cannot compute attribution",
                request_id,
                len(samples),
            )
            energy_joules = 0.0
        else:
            energy_joules = self._compute_energy(
                samples, request_id, start_time, prefill_tokens, decode_tokens
            )

        cost = energy_joules / 3_600_000 * self.ENERGY_COST_PER_KWH

        # Store energy back into registry for batch-level conservation check
        with self._registry_lock:
            if request_id in self._registry:
                self._registry[request_id]["energy_joules"] = round(energy_joules, 6)

        record = {
            "request_id": request_id,
            "start_time": start_time,
            "end_time": end_time,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "energy_joules": round(energy_joules, 6),
            "cost": round(cost, 10),
            "endpoint": endpoint,
            "model": model,
            "anomaly_flag": 0,
            "prompt_preview": prompt_preview,
        }

        if self._db_conn is not None:
            insert_request(self._db_conn, record)

        logger.info(
            "request %s: %.4f J, USD %.10f (%d prefill + %d decode tokens, %d samples)",
            request_id,
            energy_joules,
            cost,
            prefill_tokens,
            decode_tokens,
            len(samples),
        )

        return {
            "request_id": request_id,
            "energy_joules": record["energy_joules"],
            "cost": record["cost"],
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
        }

    def attribute_async(
        self,
        request_id: str,
        start_time: float,
        end_time: float,
        prefill_tokens: int,
        decode_tokens: int,
        endpoint: str,
        model: str,
        prompt_preview: str = "",
    ) -> threading.Thread:
        """Run attribution in a background thread so the proxy isn't blocked."""
        t = threading.Thread(
            target=self.attribute,
            args=(request_id, start_time, end_time, prefill_tokens, decode_tokens, endpoint, model, prompt_preview),
            daemon=True,
        )
        t.start()
        return t

    def _compute_energy(
        self,
        samples: list[tuple[float, float]],
        request_id: str,
        start_time: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> float:
        """Trapezoidal integration with weighted token share via registry."""
        my_weighted = prefill_tokens * self.W_PREFILL + decode_tokens * self.W_DECODE

        # Snapshot the registry (only finalized requests)
        with self._registry_lock:
            registry_snapshot = {
                rid: dict(info)
                for rid, info in self._registry.items()
                if info["end_time"] is not None
            }

        energy_joules = 0.0

        for i in range(len(samples) - 1):
            t1, w1 = samples[i]
            t2, w2 = samples[i + 1]
            interval = t2 - t1
            attributable_watts = max(0.0, (w1 + w2) / 2 - self.IDLE_POWER)

            midpoint = (t1 + t2) / 2

            # Find all requests active at this midpoint
            total_weighted = 0.0
            for rid, info in registry_snapshot.items():
                if info["start_time"] <= midpoint <= info["end_time"]:
                    total_weighted += info["weighted_tokens"]

            # Fallback: if ourselves not in snapshot, add ourselves
            if request_id not in registry_snapshot:
                if start_time <= midpoint <= (start_time + 999):
                    total_weighted += my_weighted

            if total_weighted > 0:
                share = my_weighted / total_weighted
            else:
                share = 1.0

            energy_joules += attributable_watts * interval * share

        return energy_joules

    def validate_energy_conservation(
        self,
        request_ids: list[str],
        global_start: float,
        global_end: float,
    ) -> dict:
        """Batch-level energy conservation check.

        Compares:
          A = sum of energy_joules attributed to all requests (from registry)
          B = total attributable GPU energy in the time window (from PowerBuffer)

        Error < 5% → PASS.
        """
        # A: sum attributed energy from registry
        total_attributed = 0.0
        with self._registry_lock:
            for rid in request_ids:
                if rid in self._registry:
                    total_attributed += self._registry[rid].get("energy_joules", 0.0)

        # B: integrate total attributable energy from PowerBuffer
        samples = self._buffer.query(global_start, global_end + self.WAIT_AFTER_END)
        total_attributable = 0.0
        for i in range(len(samples) - 1):
            t1, w1 = samples[i]
            t2, w2 = samples[i + 1]
            interval = t2 - t1
            total_attributable += max(0.0, (w1 + w2) / 2 - self.IDLE_POWER) * interval

        error_pct = 0.0
        if total_attributable > 0:
            error_pct = abs(total_attributed - total_attributable) / total_attributable * 100

        passed = error_pct < 5.0

        if not passed:
            logger.warning(
                "Energy conservation FAIL: attributed=%.4fJ, "
                "attributable=%.4fJ, error=%.2f%%",
                total_attributed, total_attributable, error_pct,
            )
        else:
            logger.info(
                "Energy conservation PASS: attributed=%.4fJ, "
                "attributable=%.4fJ, error=%.2f%%",
                total_attributed, total_attributable, error_pct,
            )

        return {
            "total_attributed_joules": round(total_attributed, 6),
            "total_attributable_joules": round(total_attributable, 6),
            "error_pct": round(error_pct, 4),
            "passed": passed,
            "sample_count": len(samples),
            "request_count": len(request_ids),
        }
