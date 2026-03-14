# PowerDecode 交戰守則 v2

> **時間軸**: 週五 5pm 上線 → 週六全天校準驗證 → 週日 11:30am Hacking → 週日 5pm 評審
> **更新**: 2026-03-13 週五 1pm 自檢通過，所有代碼到位

---

## 自檢狀態（週五 1pm）

| 項目 | 狀態 |
|------|------|
| proxy.py | ✅ |
| attribution_engine.py | ✅ |
| db.py | ✅ |
| dashboard.py | ✅ |
| collectors/gpu_power.py | ✅ (含 MultiGpuPowerSamplerNVML) |
| cluster/diagnose.py | ✅ |
| cluster/calibrate.py | ✅ |
| cluster/validate_all.py | ✅ |
| backup_remote.sh / restore_local.sh | ✅ |
| 驗證 1/2/3 腳本 | ✅ |

**已知風險牢記**:
1. `finalize_request()` 必須在 `attribute()` 之前（並發時：全部 finalize → 全部 attribute）
2. 第一筆 request 能量偏低 ~40%（warm-up），demo 前送 3 個 warm-up
3. 4060 Ti 參考常數：IDLE=21.07W, W_P=0.0212, W_D=0.1772, ratio=8.3x

---

## 週五 5:00pm — SSH 上線（目標 30 分鐘搞定）

### 5:00-5:05 | 環境確認

```bash
nvidia-smi
python3 --version       # 需 ≥ 3.9
pip install pynvml httpx fastapi uvicorn streamlit altair
```

**→ 結果判斷**:
- nvidia-smi 正常 → 繼續
- nvidia-smi 失敗 → **停下**，找主辦方確認 GPU driver，這是 blocker

### 5:05-5:10 | Clone + 診斷

```bash
# rsync 或 git clone 代碼到機器上
python3 cluster/diagnose.py
cat cluster/diagnose_result.json
```

**→ 記錄這些值**（後面每一步都要用）:

```
gpu_model:           _______________
gpu_count:           ___
gpu_memory_gb:       ___
pynvml_available:    ___
recommended_strategy: _______________
sampling_ms:         ___
```

### 5:10-5:20 | 根據診斷走分支

```
IF pynvml_available == true:
│   ✅ 主力方案，不需改代碼
│   確認 sampling_ms ≤ 10ms → OK
│
│   IF gpu_count == 1:
│   │   單卡，現有代碼直接用
│   │
│   IF gpu_count > 1:
│       確認 vLLM 是 tensor parallel 還是多實例？
│       │
│       ├─ Tensor parallel:
│       │   改 proxy.py 一行:
│       │     GpuPowerSamplerNVML(interval_ms=10)
│       │     → MultiGpuPowerSamplerNVML(interval_ms=10)
│       │   IDLE_POWER 之後用 calibrate.py 自動處理
│       │
│       └─ 多實例:
│           每卡一個 proxy 實例，不用改代碼
│
IF pynvml_available == false:
    ├─ DCGM available → 加 DCGM sampler (100ms)，改 collectors/gpu_power.py
    └─ DCGM 也沒有 → nvidia-smi fallback，Demo 避開 prefill 精度話題
```

### 5:20-5:30 | 確認 vLLM

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

**→ 結果判斷**:
- 回傳正常 → 記下 model id: `_______________`，今天任務完成
- 連不上 → vLLM 可能還沒跑，問主辦方，或等週六再確認
- OOM → `gpu_memory_utilization=0.70`, `max_num_seqs=16`

**→ 週五結束時你手上要有**: diagnose_result.json + model id

---

## 週六 全天 — 校準 + 驗證（目標 60-90 分鐘）

### 前置確認（2 分鐘）

```bash
curl -s http://localhost:8000/v1/models   # vLLM 在跑？
cat cluster/diagnose_result.json          # 昨天的診斷結果
```

- vLLM 沒在跑 → **先啟動 vLLM**，這是一切的前提

---

### 週六 Step 1 | 校準常數（~30 分鐘）

```bash
python3 cluster/calibrate.py
```

**→ 結果記錄**:

```
IDLE_POWER = ___ W
W_PREFILL  = ___ J/token
W_DECODE   = ___ J/token
```

**→ 立即做關鍵判斷 — W_DECODE / W_PREFILL 比例**:

```
ratio = W_DECODE / W_PREFILL = ___x

IF ratio > 3x:
│   ✅ 假設成立，原定 demo 方向繼續
│   Demo 論點②：「Decode token 電耗是 prefill 的 Xx」
│
IF 1x < ratio ≤ 3x:
│   ⚠️ 比例存在但偏低
│   → 調整話術：
│     "4060 Ti 上是 8.3x，H100 上是 Xx
│      不同硬體比例不同，這正是為什麼需要 calibrate，不能用假設"
│
IF ratio ≈ 1x:
    🔴 假設不成立，切換備案 B
    → 核心訊息改成：
      "你知道你的每一個 API call 花了多少電嗎？
       業界沒有工具做到 per-request 成本可視化。
       PowerDecode 做到了，而且在 64 concurrent 下仍然準確。"
    → 強調: (1) per-request 可視化 (2) concurrent attribution 正確性
```

**→ 如果 W_PREFILL 三輪差距 > 30%**:
- H100 prefill 太快（< 100ms），採樣點不夠
- 解法：改 `calibrate.py` prompt 為 3200 tokens，或 ROUNDS 改成 5
- 重跑 `python3 cluster/calibrate.py`

**→ 校準完成，手動更新**:

```bash
vim attribution_engine.py
# 改 line 94-96 的 IDLE_POWER, W_PREFILL, W_DECODE
```

**→ 如果 ratio 變了，同步更新 dashboard.py**:
- 找到 `STATS_CONTEXT` 裡的 `"Known: decode token costs 8.3x more electricity than prefill token."`
- 改成實測數字
- 如果備案 B，三個 Claude prompt 核心假設需重寫

---

### 週六 Step 2 | 跑驗證一二三（~20 分鐘）

```bash
python3 cluster/validate_all.py
```

**→ 結果判斷**:

```
驗證一（單 request 閉環）:    error < 10%  → PASS / FAIL
驗證二（並發比例分配）:       error < 15%  → PASS / FAIL
驗證三（能量守恆）:           error < 5%   → PASS / FAIL

IF 全部 PASS:
│   ✅ 進入 Step 3 端到端測試
│
IF 驗證一 FAIL:
│   → 檢查 sampler 採樣頻率是否太慢
│   → 重跑 diagnose.py 確認 sampling_ms
│   → 多卡？確認 sampler 在正確 GPU 上
│   → 修完重跑 validate_all.py
│
IF 驗證二 FAIL:
│   → 90% 是常數沒校準好
│   → 重跑 calibrate.py
│   → 確認 finalize_request() 在 attribute() 之前
│   → 修完重跑 validate_all.py
│   → 還是 FAIL → 降低門檻到 20%，Demo 時說明
│
IF 驗證三 FAIL:
    → IDLE_POWER 偏差太大
    → 重測 idle（讓 GPU 空閒 20 秒取平均）
    → 修完重跑 validate_all.py
```

---

### 週六 Step 3 | 端到端測試（~10 分鐘）

```bash
# Terminal 1
python3 proxy.py

# Terminal 2: 送 3 個 request，間隔 2s
for i in 1 2 3; do
  curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
  sleep 2
done

# 驗證
curl -s http://localhost:8001/health
curl -s http://localhost:8001/stats/recent | python3 -m json.tool
```

**→ 結果判斷**:

```
IF /health 回傳 {"status":"ok","vllm":"ok"}
   AND DB 有 3 筆 record
   AND energy_joules > 0, cost_usd > 0
   AND 第 2、3 筆 energy 數值穩定:
│   ✅ 端到端通過，進入 Step 4
│
IF /health 失敗:
│   → 確認 vLLM port 8000 在跑
│   → 確認 proxy.py 沒有 crash（看 Terminal 1 的 log）
│
IF energy_joules == 0:
│   → sampler 沒啟動，檢查 pynvml import
│   → 看 proxy.py 啟動 log 有無 error
│
IF 第 2、3 筆仍不穩定（差異 > 30%）:
    → 可能是 GPU 有其他負載
    → 用 nvidia-smi 檢查是否有別的 process
```

---

### 週六 Step 4 | 備份 Demo 數據

```bash
# 在 Fluidstack 上
bash cluster/backup_remote.sh
# 確認輸出 ≥ 30 筆、有 anomaly 數據

# 在本機（scp 指令腳本會印出）
bash cluster/restore_local.sh ~/pdd_demo_backup.db

# 本機驗證
streamlit run dashboard.py
```

---

### 週六 Step 5 | 記錄數據（週日要用）

```
硬體：_____________ x ___
IDLE_POWER = ___ W
W_PREFILL  = ___ J/token
W_DECODE   = ___ J/token
Ratio      = ___x

驗證一 error: ___%  PASS/FAIL
驗證二 error: ___%  PASS/FAIL
驗證三 error: ___%  PASS/FAIL
```

**→ 週六結束時你手上要有**:
1. 三個校準常數（已填入 attribution_engine.py）
2. 三個驗證結果（全 PASS）
3. ratio 數字（Demo 論點②）
4. 備份 DB 在本機
5. 明確知道走主力方案 / 備案 A / 備案 B

---

## 週日 11:30am — Hacking 開始（目標 30 分鐘上線）

### 11:30-11:35 | 確認環境

```bash
curl -s http://localhost:8000/v1/models     # vLLM 在跑？
cat cluster/diagnose_result.json            # 硬體資訊
```

- vLLM 沒在跑 → **先啟動 vLLM**，blocker

### 11:35-11:40 | 填常數 + 清 DB

```bash
vim attribution_engine.py
# 填入週六校準的 IDLE_POWER, W_PREFILL, W_DECODE

rm -f data/powerdecode.db     # 乾淨 Demo
```

### 11:40-11:45 | 啟動 proxy + warm-up

```bash
python3 proxy.py &

# 送 3 個 warm-up（消除 GPU 冷啟動效應，這 3 筆數據不準是正常的）
for i in 1 2 3; do
  curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"warmup"}],"max_tokens":10}'
  sleep 1
done
```

### 11:45-11:50 | 快速驗證

```bash
python3 cluster/validate_all.py
```

```
IF 全 PASS:
│   ✅ 繼續啟動 Dashboard
│
IF FAIL:
│   → 常數是否跟週六一致？
│   → 重跑 calibrate.py（快的話 10 分鐘）
│   → 再跑 validate_all.py
│   → 還是 FAIL → 用週六的備份 DB 直接 demo，不做 live
```

### 11:50-12:00 | 啟動 Dashboard + 塞數據

```bash
streamlit run dashboard.py --server.port 8501 --server.headless true &

# 送不同長度的 request，讓 Dashboard 有資料
curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Explain quantum computing in detail"}],"max_tokens":300}'

curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Write a Python quicksort"}],"max_tokens":200}'

curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'

# 多送幾輪，目標 DB 有 10+ 筆
```

### 12:00 | Demo 就緒最終確認

```
□ proxy 在跑（port 8001）
□ dashboard 在跑（port 8501）
□ DB 有 10+ 筆 request
□ Dashboard Overview 頁面有數據、有 cost_usd 數字
□ /health 回傳 {"status":"ok","vllm":"ok"}
□ 瀏覽器 http://localhost:8501 正常顯示
```

**→ 全部 OK → 等 5pm Demo**
**→ Dashboard 掛了 → `pip install --upgrade streamlit`，或直接用 `/stats/recent` API 展示**

---

## 週日 5:00pm — Demo 論點

| # | 論點 | 關鍵數字來源 |
|---|------|-------------|
| 1 | Lighthouse 告訴你叢集健不健康，PowerDecode 告訴你錢花在哪 | — |
| 2 | Decode token 電耗是 prefill 的 **___x** | 週六 calibrate ratio |
| 3 | 功耗非線性 → 加權分配才是最公平的計費方式 | 驗證數據 |
| 4 | 這份數據是業界第一份推理成本基準報告的原始材料 | — |

如果走備案 B（ratio ≈ 1x），改強調:
1. Per-request 成本可視化（業界沒有）
2. Concurrent attribution 正確性（stress test 數據佐證）

---

## 緊急應對速查表

| 狀況 | 第一反應 | 如果還是不行 |
|------|---------|-------------|
| pynvml 裝不了 | `pip install nvidia-ml-py` | nvidia-smi fallback，避開 prefill 話題 |
| vLLM OOM | `gpu_memory_utilization=0.70`, `max_num_seqs=16` | 換更小模型 |
| 採樣太慢 (>20ms) | 調高 interval | Demo 避開 prefill 精度話題 |
| 驗證全 FAIL | 重新校準 IDLE_POWER | 用備份 DB 做靜態 demo |
| 多卡不知道怎麼改 | 先用單卡跑 Demo | 多卡歸因當 future work |
| Dashboard 打不開 | `pip install --upgrade streamlit` | 直接用 `/stats/recent` API |
| DB 被鎖 | `rm data/powerdecode.db` + 重啟 proxy | — |
| W_PREFILL 三輪差距 >30% | 改 prompt 為 3200 tokens 重校準 | ROUNDS 改 5 |
| SSH 斷線 | tmux/screen 保護所有 process | — |

---

## Shell Scripts 速查

| Script | Where | Usage |
|--------|-------|-------|
| `./start.sh` | Any machine | Start vLLM + proxy + dashboard. Auto-detects running vLLM. |
| `./start.sh --warmup` | Any machine | Same as above + 3 warm-up requests to stabilize GPU power readings. |
| `./start.sh --stress` | Local (4060 Ti) | Same + run single-GPU stress test (`cluster/stress_test.py`). |
| `./start.sh --stress=h100` | Fluidstack | Same + run H100 stress test (`cluster/stress_test_h100.py`). |
| `./start.sh --warmup --stress` | Any machine | All of the above combined. |
| `scripts/start_vllm.sh` | Local (4060 Ti) | Standalone vLLM launcher. Fixed params: `gpu_mem=0.80, max_seqs=32, max_model_len=2048`. |
| `scripts/reseed.sh` | Any machine | Delete DB + re-seed ~69 demo requests via `cluster/seed_demo_data.py`. Requires proxy running. |
| `cluster/backup_remote.sh` | Fluidstack | Backup DB with timestamp. Checks ≥30 rows + anomaly data. Prints SCP command. |
| `cluster/restore_local.sh <file>` | Local | Restore DB from backup. Backs up existing DB first. Checks deps + API key. |

**Environment variables for `start.sh`:**

```
PDD_MODEL=Qwen/Qwen2.5-3B-Instruct   # Model to serve
PDD_GPU_UTIL=0.85                      # GPU memory utilization
PDD_MAX_SEQS=32                        # Max concurrent sequences
PDD_MAX_MODEL_LEN=32768                # Max context length
PDD_VLLM_PORT=8000                     # vLLM port
PDD_PROXY_PORT=8001                    # Proxy port
PDD_DASH_PORT=8501                     # Dashboard port
```

**Common combos:**

```bash
# Demo quick start (most common)
./start.sh --warmup

# Full reset + fresh demo data
bash scripts/reseed.sh

# Backup H100 data to local machine
bash cluster/backup_remote.sh           # on Fluidstack
bash cluster/restore_local.sh ~/pdd_demo_backup.db  # on local
```
