# PowerDecode

> GPU watts → per-request cost. The missing analytics layer for LLM inference.

🎥 Demo Video: [coming soon]
📊 Live Dashboard: https://powerdecode.streamlit.app
📦 GitHub: https://github.com/oneway234/PowerDecode

---

## The Problem

GPU cloud providers sell compute by the hour. But inside every inference request, two fundamentally different operations are happening:

- **Prefill**: Process the input prompt (compute-bound, fast)
- **Decode**: Generate output tokens one by one (memory-bound, slow, expensive)

Under flat per-token pricing, these two operations are billed identically. They are not identical.

---

## What We Found

Decode tokens consume **10–30× more electricity** than prefill tokens — yet flat per-token pricing treats them identically.

Dataset: 1,304 measured inference requests across 4 model sizes (3B/7B/32B/72B) on NVIDIA B200.

| Model | W_DECODE (J/token) | vs 3B |
|-------|-------------------|-------|
| Qwen2.5-3B | 0.485 | 1× |
| Qwen2.5-7B | 0.969 | 2× |
| Qwen2.5-32B | 6.208 | 12.8× |
| Qwen2.5-72B | 15.159 | 31.2× |

This asymmetry is consistent across all model sizes. It is not an artifact of any single model. It is a structural property of transformer inference.

---

## The Dashboard

PowerDecode is a proxy layer that sits between your application and vLLM. Every inference request is measured in real time:

- 10ms GPU power sampling via pynvml
- Per-request energy attribution using trapezoidal integration
- Weighted token share for concurrent request attribution
- AI-powered pricing, operations, and cost analysis (3 perspectives)

**[→ Live Dashboard](https://powerdecode.streamlit.app)**

---

## Key Visualization

The Decode Electricity Cost vs Token Share chart tells the story:

- X-axis: decode tokens (% of tokens)
- Y-axis: decode electricity (% of GPU energy)
- Diagonal: fair pricing line

If pricing reflected electricity cost, workloads would lie on the diagonal.
Instead, most workloads fall above it — decode-heavy users consume more electricity than they pay for under flat token pricing.

---

## Validation (NVIDIA B200)

| Validation | Result |
|-----------|--------|
| V1: Single request closed-loop | ✅ 0.0% error |
| V2: Concurrent attribution | ✅ <5% error |
| V3: Energy conservation | ✅ 0.3% error |
| V4: Power linearity | Observed non-linearity (expected under vLLM continuous batching — validates weighted attribution approach) |

---

## Architecture

```
Your App
    ↓
PowerDecode Proxy (port 8001)
    ↓ measures GPU power at 10ms intervals
vLLM (port 8000)
    ↓
NVIDIA B200
```

---

## How to Run

```bash
# Clone
git clone https://github.com/oneway234/PowerDecode.git
cd PowerDecode

# Install
pip install -r requirements.txt

# Set environment
export PDD_DB_PATH=./data/demo.db
export ANTHROPIC_API_KEY=your_key

# Run dashboard
streamlit run dashboard.py
```

---

## Built at SemiAnalysis × Fluidstack GTC Hackathon

March 2026 · San Jose, CA
