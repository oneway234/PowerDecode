"""SQLite storage for PowerDecode request cost records."""

import datetime
import logging
import os
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("/home/wei/PowerDecode/data/powerdecode.db")
LOG_PATH = "/home/wei/PowerDecode/data/powerdecode.log"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS requests (
    request_id     TEXT PRIMARY KEY,
    start_time     REAL,
    end_time       REAL,
    prefill_tokens INTEGER,
    decode_tokens  INTEGER,
    energy_joules  REAL,
    cost           REAL,
    endpoint       TEXT,
    model          TEXT,
    anomaly_flag   INTEGER DEFAULT 0,
    prompt_preview TEXT DEFAULT ''
);
"""


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create the database and table if they don't exist. Return a connection."""
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()

    # Migration: add prompt_preview to existing databases
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN prompt_preview TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    logger.info("Database initialized at %s", db_path)
    return conn


def compute_anomaly_flag(conn: sqlite3.Connection, request_id: str) -> int:
    """Compute anomaly_flag for a single request.

    Returns:
        0 = OK
        1 = Statistical anomaly (energy/weighted_token > mean + 2σ)
        2 = Extreme anomaly (energy_joules = 0)
    """
    W_PREFILL = 0.0212
    W_DECODE = 0.1772

    # energy=0 → extreme anomaly
    cur = conn.execute(
        "SELECT energy_joules, prefill_tokens, decode_tokens FROM requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()

    if not cur:
        return 0

    if cur["energy_joules"] == 0:
        return 2

    # 原有 mean + 2σ 邏輯，超過 → 1
    rows = conn.execute("""
        SELECT energy_joules, prefill_tokens, decode_tokens
        FROM requests
        WHERE request_id != ?
        ORDER BY end_time DESC
        LIMIT 50
    """, (request_id,)).fetchall()

    if len(rows) < 10:
        return 0

    ratios = []
    for r in rows:
        w = r["prefill_tokens"] * W_PREFILL + r["decode_tokens"] * W_DECODE
        if w > 0:
            ratios.append(r["energy_joules"] / w)

    if len(ratios) < 10:
        return 0

    mean = statistics.mean(ratios)
    std = statistics.stdev(ratios)

    if std == 0:
        return 0

    w = cur["prefill_tokens"] * W_PREFILL + cur["decode_tokens"] * W_DECODE
    if w <= 0:
        return 0

    cur_ratio = cur["energy_joules"] / w
    return 1 if cur_ratio > mean + 2 * std else 0


def insert_request(conn: sqlite3.Connection, record: dict) -> None:
    """Insert a single request record."""
    conn.execute(
        """
        INSERT OR REPLACE INTO requests
            (request_id, start_time, end_time, prefill_tokens, decode_tokens,
             energy_joules, cost, endpoint, model, anomaly_flag, prompt_preview)
        VALUES
            (:request_id, :start_time, :end_time, :prefill_tokens, :decode_tokens,
             :energy_joules, :cost, :endpoint, :model, :anomaly_flag, :prompt_preview)
        """,
        {
            "request_id": record["request_id"],
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "prefill_tokens": record["prefill_tokens"],
            "decode_tokens": record["decode_tokens"],
            "energy_joules": record["energy_joules"],
            "cost": record["cost"],
            "endpoint": record.get("endpoint", ""),
            "model": record.get("model", ""),
            "anomaly_flag": record.get("anomaly_flag", 0),
            "prompt_preview": record.get("prompt_preview", ""),
        },
    )
    conn.commit()

    # Compute anomaly flag based on historical data
    try:
        anomaly = compute_anomaly_flag(conn, record["request_id"])
        if anomaly:
            conn.execute(
                "UPDATE requests SET anomaly_flag = ? WHERE request_id = ?",
                (anomaly, record["request_id"]),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to compute anomaly flag: %s", e)

    # Append human-readable log line
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        ts = datetime.datetime.fromtimestamp(record["end_time"]).strftime("%Y-%m-%d %H:%M:%S")
        short_id = record["request_id"][:8]
        model = record.get("model", "unknown").split("/")[-1]
        latency = record["end_time"] - record["start_time"]
        line = (
            f"{ts} | req={short_id} | model={model} | "
            f"prefill={record['prefill_tokens']}tok | decode={record['decode_tokens']}tok | "
            f"latency={latency:.2f}s | energy={record['energy_joules']:.2f}J | "
            f"cost=USD {record['cost']:.8f} | anomaly={record.get('anomaly_flag', 0)}\n"
        )
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception as e:
        logger.warning("Failed to write log: %s", e)


def get_recent_requests(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    """Return the most recent N requests, ordered by end_time descending."""
    cursor = conn.execute(
        "SELECT * FROM requests ORDER BY end_time DESC LIMIT ?",
        (limit,),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_requests_by_timerange(
    conn: sqlite3.Connection, start: float, end: float
) -> list[dict]:
    """Return all requests whose end_time falls within [start, end]."""
    cursor = conn.execute(
        "SELECT * FROM requests WHERE end_time >= ? AND end_time <= ? ORDER BY end_time",
        (start, end),
    )
    return [dict(row) for row in cursor.fetchall()]
