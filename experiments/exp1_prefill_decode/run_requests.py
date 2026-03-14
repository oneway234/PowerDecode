#!/usr/bin/env python3
"""
Experiment 1: Prefill vs Decode power consumption measurability.

Three test groups sent to vLLM via OpenAI-compatible API:
  A) Pure Prefill  — long prompt (2000+ tokens), max_tokens=1
  B) Pure Decode   — short prompt (<10 tokens), max_tokens=512
  C) Mixed Baseline — medium prompt (~200 tokens), max_tokens=200

Each group runs 5 rounds with 5-second idle gaps between rounds.
Results are saved to experiments/exp1_prefill_decode/results/request_log.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000/v1"


def _detect_model() -> str:
    try:
        resp = httpx.get(f"{API_BASE_URL}/models", timeout=10.0)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "Qwen/Qwen2.5-3B-Instruct"


MODEL_NAME = _detect_model()
NUM_ROUNDS = 5
IDLE_SLEEP_SECONDS = 5

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = RESULTS_DIR / "request_log.json"

# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

# A long English paragraph (~80 tokens) to be repeated for the long prompt.
_FILLER_PARAGRAPH = (
    "The rapid advancement of large language models has transformed the landscape "
    "of artificial intelligence research and application development. These models, "
    "trained on vast corpora of text data, demonstrate remarkable capabilities in "
    "natural language understanding, generation, and reasoning. Researchers continue "
    "to explore novel architectures, training methodologies, and alignment techniques "
    "to improve both the efficiency and safety of these systems. The computational "
    "cost associated with training and serving these models remains a significant "
    "consideration for organizations deploying them at scale. Understanding the "
    "energy consumption patterns during inference is crucial for accurate cost "
    "attribution and sustainable deployment practices. "
)


def _make_long_prompt() -> str:
    """Build a long prompt by repeating a filler paragraph."""
    # Each repetition is ~80 tokens; 15 repetitions -> ~1200 tokens.
    # Qwen chat template adds significant overhead; leave headroom for max_model_len=2048.
    return _FILLER_PARAGRAPH * 15


def _make_short_prompt() -> str:
    """A short but instruction-style prompt (<30 tokens) that elicits long output."""
    return (
        "Write a detailed step-by-step tutorial on how to build "
        "a REST API with Python Flask. Include code examples for "
        "each step, explain authentication, database integration, "
        "error handling, and deployment. Be thorough and "
        "comprehensive."
    )


def _make_medium_prompt() -> str:
    """Build a prompt of roughly 200 tokens."""
    # 3 repetitions -> ~240 tokens.
    return _FILLER_PARAGRAPH * 3


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS: list[dict] = [
    {
        "name": "A_pure_prefill",
        "description": "Long prompt (2000+ tokens), max_tokens=1",
        "prompt_fn": _make_long_prompt,
        "max_tokens": 1,
    },
    {
        "name": "B_pure_decode",
        "description": "Short prompt (<30 tokens), max_tokens=600",
        "prompt_fn": _make_short_prompt,
        "max_tokens": 600,
    },
    {
        "name": "C_mixed_baseline",
        "description": "Medium prompt (~200 tokens), max_tokens=200",
        "prompt_fn": _make_medium_prompt,
        "max_tokens": 200,
    },
]

# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


def send_request(
    client: OpenAI,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send a single chat-completion request and return timing/token metadata."""
    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    end_time = time.time()

    usage = response.usage
    return {
        "start_time": start_time,
        "end_time": end_time,
        "latency": end_time - start_time,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all_tests() -> list[dict]:
    """Execute all test groups and return the full log."""
    client = OpenAI(
        api_key="EMPTY",  # vLLM does not require a real key
        base_url=API_BASE_URL,
    )

    all_results: list[dict] = []

    for test in TESTS:
        prompt = test["prompt_fn"]()
        print(f"\n{'='*60}")
        print(f"Test: {test['name']} — {test['description']}")
        print(f"{'='*60}")

        for round_idx in range(1, NUM_ROUNDS + 1):
            print(f"  Round {round_idx}/{NUM_ROUNDS} ... ", end="", flush=True)

            result = send_request(client, prompt, test["max_tokens"])
            result["test_name"] = test["name"]
            result["round"] = round_idx
            all_results.append(result)

            print(
                f"latency={result['latency']:.3f}s  "
                f"prompt_tokens={result['prompt_tokens']}  "
                f"completion_tokens={result['completion_tokens']}"
            )

            # Sleep between rounds to let GPU return to idle baseline.
            if round_idx < NUM_ROUNDS:
                time.sleep(IDLE_SLEEP_SECONDS)

        # Also sleep between different test groups.
        print(f"  Sleeping {IDLE_SLEEP_SECONDS}s before next test group ...")
        time.sleep(IDLE_SLEEP_SECONDS)

    return all_results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_all_tests()

    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for test in TESTS:
        name = test["name"]
        entries = [r for r in results if r["test_name"] == name]
        latencies = [r["latency"] for r in entries]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_prompt = (
            sum(r["prompt_tokens"] for r in entries if r["prompt_tokens"])
            / len(entries)
            if entries
            else 0
        )
        avg_completion = (
            sum(r["completion_tokens"] for r in entries if r["completion_tokens"])
            / len(entries)
            if entries
            else 0
        )
        print(
            f"  {name}: avg_latency={avg_latency:.3f}s  "
            f"avg_prompt_tokens={avg_prompt:.0f}  "
            f"avg_completion_tokens={avg_completion:.0f}"
        )


if __name__ == "__main__":
    main()
