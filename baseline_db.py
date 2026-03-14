"""PowerDecode Baseline Database — track W_PREFILL / W_DECODE across model × GPU combos."""

import datetime
import json
from pathlib import Path

BASELINES_PATH = Path(__file__).resolve().parent / "data" / "benchmark_baselines.json"


def _load() -> dict:
    if not BASELINES_PATH.exists():
        return {"version": "1.0", "updated_at": "", "baselines": []}
    return json.loads(BASELINES_PATH.read_text())


def _save(data: dict) -> None:
    data["updated_at"] = datetime.date.today().isoformat()
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(data, indent=2) + "\n")


def get_baseline(model: str, gpu: str) -> dict | None:
    """Find baseline by exact model + gpu match."""
    data = _load()
    for b in data["baselines"]:
        if b["model"] == model and b["gpu"] == gpu:
            return b
    return None


def upsert_baseline(
    model: str,
    gpu: str,
    idle_power_w: float,
    w_prefill: float,
    w_decode: float,
    measured_by: str = "powerdecode-calibrate",
    notes: str = "",
) -> None:
    """Update or insert a baseline entry."""
    data = _load()
    for b in data["baselines"]:
        if b["model"] == model and b["gpu"] == gpu:
            b["idle_power_w"] = idle_power_w
            b["w_prefill"] = w_prefill
            b["w_decode"] = w_decode
            b["measured_by"] = measured_by
            if notes:
                b["notes"] = notes
            _save(data)
            return

    data["baselines"].append({
        "model": model,
        "gpu": gpu,
        "idle_power_w": idle_power_w,
        "w_prefill": w_prefill,
        "w_decode": w_decode,
        "measured_by": measured_by,
        "notes": notes or "auto-added",
    })
    _save(data)


def list_baselines() -> list[dict]:
    """Return all baselines."""
    return _load()["baselines"]


def print_table() -> None:
    """Print human-readable comparison table."""
    baselines = list_baselines()
    if not baselines:
        print("No baselines found.")
        return

    header = f"{'Model':<30} | {'GPU':<22} | {'idle':>6} | {'W_pre':>7} | {'W_dec':>7} | Status"
    print(header)
    print("-" * len(header))
    for b in baselines:
        model_short = b["model"].split("/")[-1]
        idle = f"{b['idle_power_w']:.2f}" if b["idle_power_w"] is not None else "-"
        wp = f"{b['w_prefill']:.4f}" if b["w_prefill"] is not None else "-"
        wd = f"{b['w_decode']:.4f}" if b["w_decode"] is not None else "-"
        status = "measured" if b["w_prefill"] is not None else "pending"
        print(f"{model_short:<30} | {b['gpu']:<22} | {idle:>6} | {wp:>7} | {wd:>7} | {status}")


if __name__ == "__main__":
    print_table()
