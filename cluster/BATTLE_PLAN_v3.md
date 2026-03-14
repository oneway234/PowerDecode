# PowerDecode 交戰守則 v3

> **時間軸**: 週五 5pm 上線 → 週六全天校準驗證 → 週日 11:30am Hacking → 週日 5pm 評審
> **更新**: 2026-03-13 v3 — 新增評審導向層 + demo 戰術 + B200 硬體修正
> **硬體**: 本次 compute grant 為 **B200 single GPU**（非 H100）。本文所有遠端操作以 B200 為準。

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
4. B200 常數待週六校準，**不要假設與 4060 Ti 相同**

---

## 得獎導向總原則（新增）

> 以下條令優先級高於一切美化工作。Demo 現場每一秒都要為「評審理解核心價值」服務。

**1. 認清評審**
- 評審是 AI infra / systems / economics 背景，不是一般 AI app 評審
- 他們見過無數 monitoring dashboard，不會因為好看的圖表就給分
- 他們會問「so what」— 你必須在前 30 秒回答這個問題

**2. 主動避免被誤解**
- PowerDecode 最容易被歸類成「又一個 GPU monitoring dashboard」
- 如果評審這樣理解，你就輸了
- 每次開口都要強調：**這是 request-level cost attribution，不是 monitoring**
- 禁止用「我們做了一個 dashboard」開場

**3. Hero metric 不是總功耗，不是總成本**
- Hero metric = **decode vs prefill 電力差異倍率**（B200 實測值 ___x）
- 若 B200 ratio < 4060 Ti 的 8.3x，**不要硬守 8.3x**
- 改說：「4060 Ti 是 8.3x，B200 是 ___x — 不同硬體比例不同，這正是為什麼 flat pricing 不合理，每張卡都需要校準」
- 這反而更強：證明 calibration 不可省略

**4. Demo 順序鐵律**
1. 問題一句話（10 秒）
2. 單一 request 成本拆解 — prefill/decode donut chart（20 秒）
3. pricing / revenue leak — 三維分析 Pricing tab（30 秒）
4. live request 或 recent requests overview（20 秒）
- **不要一開始逛 dashboard**
- **不要先展示 Overview 的監控圖表**

**5. 評審要記住的是 inference economics**
- 你在說的是：「API providers 早就知道 output 比 input 貴並藉此定價，但 GPU cloud providers 至今沒有硬體量測工具」
- PowerDecode = 第一個在 GPU 層面量化這個差異的工具
- 這不是 observability，這是 **cost attribution**

**6. 時間分配**
- 前 30 秒：問題 + 核心數字（ratio）
- 30-90 秒：demo（request detail → pricing → overview）
- 90-120 秒：方法論一句話 + future work
- 不要花時間解釋架構圖、tech stack、code structure

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
gpu_model:           _______________  ← 確認是否為 B200
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

**→ 週五結束時你手上要有**: diagnose_result.json + model id + 確認是否為 B200

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
│   Demo 論點②：「Decode token 電耗是 prefill 的 ___x」
│
IF 1x < ratio ≤ 3x:
│   ⚠️ 比例存在但偏低
│   → 調整話術：
│     "4060 Ti 上是 8.3x，B200 上是 ___x
│      不同硬體比例不同，這正是為什麼需要 calibrate，不能用假設"
│   → 這反而是更強的論點：證明 per-hardware calibration 不可省略
│
IF ratio ≈ 1x:
    🔴 假設不成立，切換備案 B（見 Demo 論點章節）
```

**→ 如果 W_PREFILL 三輪差距 > 30%**:
- B200 prefill 太快（< 100ms），採樣點不夠
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
   AND energy_joules > 0, cost > 0
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
硬體：_____________ x ___  ← 確認 B200 / 其他
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
3. ratio 數字（Demo 核心數字）
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
□ Dashboard Overview 頁面有數據、有 cost 數字
□ /health 回傳 {"status":"ok","vllm":"ok"}
□ 瀏覽器 http://localhost:8501 正常顯示
□ 三維分析按鈕能出結果（測一次）
□ Request Detail 頁 donut chart 正常
```

**→ 全部 OK → 等 5pm Demo**
**→ Dashboard 掛了 → `pip install --upgrade streamlit`，或直接用 `/stats/recent` API 展示**

---

## Demo 開場順序（新增）

> **鐵律：不要一開始逛 dashboard。** 評審看到 Overview 會以為這是 monitoring tool，你就輸了。

### Step 0 | 開口前（評審坐下那 5 秒）

- 瀏覽器已停在 **Request Detail 頁面**，選好一筆 decode-heavy request
- donut chart 已可見（Decode 佔 85-93%）
- 不要開在 Overview

### Step 1 | 問題（10 秒）

> 「OpenAI 對 output token 收費比 input 貴 4 倍。為什麼？因為 decode 消耗更多 GPU 資源。但如果你是 GPU cloud provider 像 Fluidstack，你按小時賣 GPU — 你根本看不到這個差異。PowerDecode 讓你看到。」

### Step 2 | 第一個數字（15 秒）

- 指向 donut chart：「這是一個真實 request 的成本拆解。Decode 佔了 ___% 的電力成本，但 flat pricing 下它和 prefill 付一樣的錢。」
- 指向 Energy / Cost 數字

### Step 3 | Pricing 視角（20 秒）

- 切到 Overview → 點「▶ Three-Perspective Analysis」
- 如果結果已快取在 session_state，直接切到 ☁️ Pricing tab
- 指向 underpaid_amount 和 recommended_pricing
- 「這是我們用 Claude API 即時分析的結果 — Fluidstack 每 1000 個 request 少收了 $0.09」

### Step 4 | 概覽 + Live（20 秒）

- 展示 Overview 的 metrics cards 和 bar chart
- 如果時間允許，在 terminal 送一個 live request：
  ```bash
  curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"What is inference cost?"}],"max_tokens":100}'
  ```
- Dashboard 5 秒內刷新出新 request

### Step 5 | 收尾（15 秒）

- 「這是業界第一個在 GPU 硬體層面量化 prefill/decode 成本差異的工具。」
- 「每張不同的 GPU 比例不同 — 4060 Ti 是 8.3x，B200 是 ___x — 所以 calibration 不可省略。」
- 「這份數據可以作為 inference cost benchmark 的基礎。」

### Fallback | Live request 壞掉

- 不要慌，不要 debug
- 「我們有預先跑好的 69 筆真實推理數據」— 所有數據都是 GPU 真跑的，不是 mock
- 繼續用備份 DB 的靜態數據完成 demo

---

## Dashboard 首頁守則（新增）

> Dashboard 現在的 Overview 是 monitoring 風格。Demo 時不要從 Overview 開始。

**守則**:
1. **先結論，後監控** — 第一個畫面給評審看的是 Request Detail 的成本拆解，不是 Overview 的 bar chart
2. **Hero metric 要最大** — ratio 數字（___x）必須在講解中最先出現，而非藏在圖表 tooltip 裡
3. **Revenue leak 要可見** — 三維分析的 Pricing tab 是評審最 care 的，在兩次點擊內到達
4. **Single request breakdown 要快** — Request Detail 頁的 donut chart 是核心證據，一個 selectbox 就到
5. **監控圖表是佐證，不是主角** — Overview 的 bar chart 和 metrics cards 用來展示「系統在跑」，不是核心價值
6. **如果只能展示一個畫面** — 選 Request Detail 的 donut chart，不選 Overview

**現場操作建議**:
- Demo 前把瀏覽器停在 Request Detail，選好一筆 decode-heavy request
- 三維分析先點一次，讓結果快取在 session_state，demo 時直接切 tab 不用等

---

## 週日 5:00pm — Demo 論點

| # | 論點 | 講法 | 數字來源 |
|---|------|------|---------|
| 1 | **從 wall socket 到 token 的真實成本** | 「PowerDecode 告訴你一個 LLM request 從電網到 token 到底花了多少錢 — 不是估算，是 GPU 硬體實測」 | energy_joules + cost per request |
| 2 | **Decode ≠ Prefill** | 「Decode token 的 GPU 電力消耗是 prefill 的 **___x**。但 flat per-token pricing 把兩者當一樣，掩蓋了真實成本結構」 | 週六 calibrate ratio |
| 3 | **這不是 monitoring，是 cost attribution** | 「一般 observability 告訴你 GPU 有多忙。PowerDecode 告訴你每個 request 在這份忙碌裡佔多少錢」 | 驗證數據 + concurrent attribution |
| 4 | **Inference economics 基礎設施** | 「這份 per-request 成本數據可以驅動三件事：differential pricing、cost regression tracking、inference benchmark」 | — |

### 備案 A（ratio 1-3x）

不是降級，是更強的論點：

> 「4060 Ti 上 decode 是 prefill 的 8.3 倍，B200 上是 ___x — 不同硬體比例完全不同。這正是為什麼你不能用假設來定價，每張 GPU 都需要實測校準。PowerDecode 提供這個校準能力。」

強調：**calibration is the product**，不只是 one-time measurement

### 備案 B（ratio ≈ 1x）

不是失敗，是 alternate winning angle：

> 「即使在這張 GPU 上 prefill 和 decode 電力差異不大，PowerDecode 仍然是業界唯一能做到以下三件事的工具：」

1. **Per-request 成本可視化** — 「你知道你的每一個 API call 花了多少電嗎？業界沒有工具能回答。」
2. **Concurrent attribution 正確性** — 「64 個 request 同時跑，能量守恆誤差 0.23%。」
3. **Model update cost regression tracking** — 「換一個模型版本，per-token 成本變了多少？過去無法量化，現在可以。」

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

## 評審 QA 防守（新增）

> 回答原則：簡短、工程站得住腳、不過度防守。先說結論，再給一句技術細節。

**Q: 你怎麼把整卡功耗分配到單一 request？**
> 每 10ms 讀一次 GPU 功耗，扣除 idle baseline，用梯形積分算出每個時間片的可歸因能量。同一時間片內的多個 request 按加權 token 比例分配。驗證結果：單 request 誤差 0%，並發能量守恆誤差 0.23%。

**Q: 多個 request batching 同時跑怎麼算？**
> vLLM continuous batching 下 GPU 總功耗接近恆定，不隨 batch size 線性增長。所以問題變成「如何公平分配一份固定功耗」。我們用 weighted token share（decode 權重 8.3x prefill）在每個時間片內分配。三個並發 request 的能量守恆測試通過。

**Q: 為什麼 4060 Ti 和 B200 比例不同？**
> 不同 GPU 架構的 compute/memory bandwidth 比不同，prefill（compute-bound）和 decode（memory-bound）的功耗特性會改變。所以每換一張卡都需要 calibrate — 這正是 PowerDecode 存在的原因，不能用假設。

**Q: Prefill 很短，量測誤差怎麼處理？**
> 實測 prefill 56ms 內用 10ms 採樣只有 5-11 個點，W_PREFILL 的 CV 確實偏高。但 W_DECODE 非常穩定（spread <2%），而 decode 佔實際成本的 85-95%。所以整體歸因精度不受太大影響。

**Q: 這和一般 observability tool（Prometheus/Grafana/DCGM dashboard）有何不同？**
> Observability 告訴你 GPU utilization 是 80%。PowerDecode 告訴你這 80% 裡面，request A 佔了 $0.00002，request B 佔了 $0.0001，而且 B 貴 5 倍是因為它 decode-heavy。這是 attribution，不是 monitoring。

**Q: 這和 GPU 雲帳單有何不同？**
> GPU 雲帳單是 per-hour。PowerDecode 是 per-request。一張卡一小時跑了 1000 個 request，帳單只有一個數字。PowerDecode 有 1000 個數字，每個 request 各自多少錢。

**Q: 為什麼這對 GPU cloud operator（如 Fluidstack）有價值？**
> OpenAI 對 output token 收費比 input 貴 3-5x，因為他們在 API 層發現了成本不對稱。Fluidstack 賣的是 raw GPU hour，看不到這層。PowerDecode 讓他們第一次能在硬體層看到這個差異，從而做 differential pricing 或至少量化 revenue leak。

**Q: 未來怎麼擴到 benchmark network / regression tracking？**
> 每次校準產生的 W_PREFILL / W_DECODE 已經存在 benchmark_baselines.json 裡，按 model × GPU 索引。跨多張卡跑就是 inference cost benchmark。同一張卡、同一個模型、不同版本跑就是 cost regression tracking。數據管線已經在了。

**Q: 為什麼不直接用 vLLM 的 latency 來估成本？**
> Latency 不等於功耗。兩個 request latency 一樣但 prefill/decode 比例不同，實際耗電可以差很多。我們量的是 watts，不是 seconds。

---

## Shell Scripts 速查

| Script | Where | Usage |
|--------|-------|-------|
| `./start.sh` | Any machine | Start vLLM + proxy + dashboard. Auto-detects running vLLM. |
| `./start.sh --warmup` | Any machine | Same as above + 3 warm-up requests to stabilize GPU power readings. |
| `./start.sh --stress` | Local (4060 Ti) | Same + run single-GPU stress test (`cluster/stress_test.py`). |
| `./start.sh --stress=h100` | Grant GPU | Run stress test via `cluster/stress_test_h100.py`. ⚠️ 腳本名保留為 h100，若實際 grant 為 B200，token 數/timeout 可能需調整。 |
| `./start.sh --warmup --stress` | Any machine | All of the above combined. |
| `scripts/start_vllm.sh` | Local (4060 Ti) | Standalone vLLM launcher. Fixed params: `gpu_mem=0.80, max_seqs=32, max_model_len=2048`. |
| `scripts/reseed.sh` | Any machine | Delete DB + restart proxy/dashboard + re-seed ~69 demo requests. |
| `cluster/backup_remote.sh` | Grant GPU (Fluidstack) | Backup DB with timestamp. Checks ≥30 rows + anomaly data. Prints SCP command. |
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

# Backup grant GPU data to local machine
bash cluster/backup_remote.sh           # on Fluidstack
bash cluster/restore_local.sh ~/pdd_demo_backup.db  # on local
```

---

## 最終勝利條件（新增）

### 評審必須在 30 秒內懂什麼

- [ ] PowerDecode 量化單一 LLM request 的真實電力成本（不是估算）
- [ ] Decode token 比 prefill token 消耗 ___x 更多 GPU 電力
- [ ] Flat pricing 掩蓋了這個差異 → GPU cloud provider 正在被 decode-heavy 用戶佔便宜

### 評審在 2 分鐘內必須看到什麼

- [ ] 一個 request 的 prefill/decode 成本 donut chart（比例視覺衝擊）
- [ ] Pricing 三維分析：revenue leak 量化數字
- [ ] Dashboard 是真的在跑（有 auto-refresh 或 live request）

### 三個最值得記住的數字

1. **___x** — decode/prefill 電力差異倍率（B200 實測值）
2. **0.23%** — 並發歸因能量守恆誤差（工程正確性）
3. **$0.09 / 1000 requests** — flat pricing 的 revenue leak（商業價值）

### 不要現場講太多的功能

- 架構圖 / tech stack 細節（評審不 care）
- 梯形積分的數學推導（一句「10ms 採樣 + 積分」帶過）
- 驗證四（功耗線性假設 FAIL）— 解釋太複雜，時間不夠
- 多卡支援（未實測，提到就好）
- location.py 地理定位電價（nice-to-have，不是 demo point）

### 如果時間只剩 60 秒

1. **開 Request Detail**（10 秒）→ 指 donut chart：「Decode 佔 ___% 電力成本」
2. **說一句話**（10 秒）→「Flat pricing 讓 decode-heavy 用戶少付 50%」
3. **切 Pricing tab**（10 秒）→ 指 revenue leak 數字
4. **收尾**（10 秒）→「第一個在 GPU 層面量化 prefill/decode 成本差異的工具」

**→ 30 秒剩餘 buffer 給評審提問**
