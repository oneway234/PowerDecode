#!/bin/bash
# PowerDecode — 一鍵啟動 vLLM + proxy + dashboard
#
# Usage:
#   ./start.sh                          # 啟動全部，不跑測試
#   ./start.sh --warmup                 # 啟動 + warmup（3 個預熱 request）
#   ./start.sh --stress                 # 啟動 + 單卡壓力測試（4060 Ti）
#   ./start.sh --stress=h100            # 啟動 + H100 壓力測試
#   ./start.sh --warmup --stress        # 啟動 + warmup + 壓力測試
#
# vLLM 參數可透過環境變數覆蓋：
#   PDD_MODEL=Qwen/Qwen2.5-3B-Instruct
#   PDD_GPU_UTIL=0.85
#   PDD_MAX_SEQS=32
#   PDD_MAX_MODEL_LEN=32768
#   PDD_VLLM_PORT=8000
#   PDD_PROXY_PORT=8001
#   PDD_DASH_PORT=8501

set -e
cd "$(dirname "$0")"

# ======================================================================
# Defaults (override with env vars)
# ======================================================================
MODEL="${PDD_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GPU_UTIL="${PDD_GPU_UTIL:-0.85}"
MAX_SEQS="${PDD_MAX_SEQS:-32}"
MAX_MODEL_LEN="${PDD_MAX_MODEL_LEN:-32768}"
VLLM_PORT="${PDD_VLLM_PORT:-8000}"
PROXY_PORT="${PDD_PROXY_PORT:-8001}"
DASH_PORT="${PDD_DASH_PORT:-8501}"

DO_WARMUP=false
STRESS_MODE=""  # "", "single", "h100"

# ======================================================================
# Parse flags
# ======================================================================
for arg in "$@"; do
    case "$arg" in
        --warmup)
            DO_WARMUP=true
            ;;
        --stress)
            STRESS_MODE="single"
            ;;
        --stress=h100)
            STRESS_MODE="h100"
            ;;
        --stress=single)
            STRESS_MODE="single"
            ;;
        --help|-h)
            head -15 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: ./start.sh [--warmup] [--stress|--stress=h100]"
            exit 1
            ;;
    esac
done

# Track PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    # Also kill vLLM if we started it
    if [ -n "$VLLM_PID" ]; then
        kill "$VLLM_PID" 2>/dev/null
    fi
    echo "Stopped."
    exit 0
}
trap cleanup INT TERM

# ======================================================================
# Step 1: vLLM
# ======================================================================
echo "========================================="
echo "  PowerDecode Startup"
echo "========================================="
echo ""

# Check if vLLM is already running
if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    EXISTING_MODEL=$(curl -s "http://localhost:${VLLM_PORT}/v1/models" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
    echo "[vLLM] Already running on port ${VLLM_PORT} — model: ${EXISTING_MODEL}"
    MODEL="$EXISTING_MODEL"
else
    echo "[vLLM] Starting on port ${VLLM_PORT}..."
    echo "  Model:         ${MODEL}"
    echo "  GPU util:      ${GPU_UTIL}"
    echo "  Max seqs:      ${MAX_SEQS}"
    echo "  Max model len: ${MAX_MODEL_LEN}"

    vllm serve "$MODEL" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs "$MAX_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        > /tmp/powerdecode_vllm.log 2>&1 &
    VLLM_PID=$!
    PIDS+=("$VLLM_PID")

    echo "[vLLM] Waiting for model to load (PID ${VLLM_PID})..."
    # Poll until vLLM is ready (up to 120s)
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            break
        fi
        # Check if process died
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM] ERROR: Process died. Check /tmp/powerdecode_vllm.log"
            tail -20 /tmp/powerdecode_vllm.log
            exit 1
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[vLLM] ERROR: Failed to start within 120s. Check /tmp/powerdecode_vllm.log"
        tail -20 /tmp/powerdecode_vllm.log
        exit 1
    fi
    echo "[vLLM] Ready."
fi
echo ""

# ======================================================================
# Step 2: Proxy
# ======================================================================
# Kill stale proxy if port is occupied
if lsof -ti:"${PROXY_PORT}" > /dev/null 2>&1; then
    echo "[Proxy] Killing stale process on port ${PROXY_PORT}..."
    lsof -ti:"${PROXY_PORT}" | xargs kill -9 2>/dev/null
    sleep 1
fi

echo "[Proxy] Starting on port ${PROXY_PORT}..."
python3 proxy.py > /tmp/powerdecode_proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=("$PROXY_PID")
sleep 3

if ! curl -s "http://localhost:${PROXY_PORT}/health" > /dev/null 2>&1; then
    echo "[Proxy] ERROR: Failed to start. Check /tmp/powerdecode_proxy.log"
    tail -20 /tmp/powerdecode_proxy.log
    cleanup
fi
echo "[Proxy] Ready (PID ${PROXY_PID})."
echo ""

# ======================================================================
# Step 3: Dashboard
# ======================================================================
# Kill stale dashboard if port is occupied
if lsof -ti:"${DASH_PORT}" > /dev/null 2>&1; then
    echo "[Dashboard] Killing stale process on port ${DASH_PORT}..."
    lsof -ti:"${DASH_PORT}" | xargs kill -9 2>/dev/null
    sleep 1
fi

echo "[Dashboard] Starting on port ${DASH_PORT}..."
streamlit run dashboard.py \
    --server.port "$DASH_PORT" \
    --server.headless true \
    > /tmp/powerdecode_dashboard.log 2>&1 &
DASH_PID=$!
PIDS+=("$DASH_PID")
sleep 2
echo "[Dashboard] Ready (PID ${DASH_PID})."
echo ""

# ======================================================================
# Step 4: Warmup (optional)
# ======================================================================
if [ "$DO_WARMUP" = true ]; then
    echo "[Warmup] Sending 3 warmup requests..."
    for i in 1 2 3; do
        RESP=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST "http://localhost:${PROXY_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}")
        if [ "$RESP" = "200" ]; then
            echo "  warmup ${i}/3 done"
        else
            echo "  warmup ${i}/3 failed (HTTP ${RESP})"
        fi
        sleep 1
    done
    echo "[Warmup] Complete."
    echo ""
fi

# ======================================================================
# Summary
# ======================================================================
echo "========================================="
echo "  PowerDecode Running"
echo "========================================="
echo "  Model:     ${MODEL}"
echo "  vLLM:      http://localhost:${VLLM_PORT}"
echo "  Proxy:     http://localhost:${PROXY_PORT}/v1/chat/completions"
echo "  Dashboard: http://localhost:${DASH_PORT}"
echo "  Health:    http://localhost:${PROXY_PORT}/health"
echo "  Logs:      /tmp/powerdecode_{vllm,proxy,dashboard}.log"
echo ""

# ======================================================================
# Step 5: Stress test (optional, runs then returns to wait)
# ======================================================================
if [ -n "$STRESS_MODE" ]; then
    echo "========================================="
    echo "  Stress Test: ${STRESS_MODE}"
    echo "========================================="
    echo ""

    if [ "$STRESS_MODE" = "h100" ]; then
        python3 cluster/stress_test_h100.py
    else
        python3 cluster/stress_test.py
    fi

    echo ""
    echo "Stress test finished. Services still running."
    echo "Press Ctrl+C to stop all services."
    echo ""
fi

# Keep foreground so Ctrl+C kills all
wait
