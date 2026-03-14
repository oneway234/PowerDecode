"""Seed middle-ground requests for decode asymmetry scatter plot."""

import asyncio

import httpx

PROXY_URL = "http://localhost:8001"
COMPLETIONS_URL = f"{PROXY_URL}/v1/chat/completions"


def detect_model() -> str:
    try:
        resp = httpx.get(f"http://localhost:8000/v1/models", timeout=10.0)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "Qwen/Qwen2.5-3B-Instruct"


REQUESTS = [
    # 極低 decode 比例（decode token % < 10%）
    ("word " * 200 + "What is the answer?", 8),
    ("word " * 300 + "Summarize in one word.", 5),
    ("word " * 400 + "Give a one-word answer.", 6),
    ("word " * 150 + "Yes or no?", 3),
    ("word " * 250 + "Name one thing.", 5),
    ("word " * 500 + "One word only.", 4),
    ("word " * 350 + "True or false?", 4),
    # 低 decode 比例（decode token % 15-30%）
    ("word " * 100 + "Explain in 20 words.", 20),
    ("word " * 80 + "Summarize briefly.", 25),
    ("word " * 120 + "Describe in one sentence.", 18),
    ("word " * 90 + "What is the key point?", 22),
    # 中間比例（decode token % 40-60%）
    ("Explain AI in 50 words.", 50),
    ("What is machine learning?", 55),
    ("Describe a GPU briefly.", 48),
    ("What is deep learning?", 52),
    ("Explain transformers briefly.", 50),
    ("What is backpropagation?", 46),
    ("Define neural network.", 50),
    ("What is overfitting?", 54),
    ("Explain gradient descent.", 48),
    ("What is a tensor?", 50),
    ("How does attention work?", 52),
    ("What is a learning rate?", 48),
    # 稍高比例（decode token % 60-75%）
    ("Tell me about AI.", 100),
    ("Explain inference in LLMs.", 90),
    ("What is CUDA?", 85),
    ("How does vLLM work?", 95),
    ("Explain KV cache.", 88),
]


async def send_request(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int) -> None:
    try:
        resp = await client.post(
            COMPLETIONS_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
        p = usage.get("prompt_tokens", 0)
        d = usage.get("completion_tokens", 0)
        pct = d / (p + d) * 100 if (p + d) > 0 else 0
        print(f"  ok | decode {pct:.0f}% | {prompt[:45]}")
    except Exception as e:
        print(f"  failed: {e}")


async def main() -> None:
    print("Detecting model...")
    model = detect_model()
    print(f"  Model: {model}\n")

    print(f"Sending {len(REQUESTS)} requests...\n")

    async with httpx.AsyncClient() as client:
        for prompt, max_tokens in REQUESTS:
            await send_request(client, model, prompt, max_tokens)
            await asyncio.sleep(0.3)

    print(f"\nDone. {len(REQUESTS)} requests sent.")


if __name__ == "__main__":
    asyncio.run(main())
