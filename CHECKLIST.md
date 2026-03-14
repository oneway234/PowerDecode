# PowerDecode 比賽前完整 Checklist

> 週五 5pm 上線 → 週六全天驗證 → 週日 11:30am Hacking → 週日 5pm 評審

---

## 【今天 Thursday】出發前確認

### 代碼完整性
- [ ] `proxy.py` — FastAPI port 8001，轉發 vLLM port 8000
- [ ] `attribution_engine.py` — PowerBuffer + AttributionEngine，常數已填（4060 Ti 值）
- [ ] `db.py` — SQLite wrapper，schema 正確
- [ ] `dashboard.py` — Streamlit port 8501
- [ ] `collectors/gpu_power.py` — GpuPowerSamplerNVML + MultiGpuPowerSamplerNVML（已驗 import OK）
- [ ] `cluster/diagnose.py` — 輸出 diagnose_result.json
- [ ] `cluster/calibrate.py` — 輸出 calibrate_result.json + copy-paste snippet
- [ ] `cluster/validate_all.py` — 一鍵跑驗證一二三，輸出 validate_result.json
- [ ] `cluster/BATTLE_PLAN.md` — 交戰守則

### 已知 Bug / 風險 確認
- [ ] `finalize_request()` 必須在 `attribute()` 之前呼叫（並發情況下對所有 request 先全部 finalize 再全部 attribute）
- [ ] warm-up 效應：第一筆 request 能量會偏低 ~40%，demo 前先送 3 個 warm-up
- [ ] MultiGpuPowerSamplerNVML 已實作並 import 驗證通過

### 帶去現場的資訊
- [ ] Fluidstack SSH 憑證 / IP
- [ ] vLLM 啟動指令（含模型路徑）
- [ ] 4060 Ti 參考常數（備查）：IDLE=21.07W, W_P=0.0212, W_D=0.1772, ratio=8.3x

---

## 【週五 5pm】SSH 上線後（目標 30 分鐘完成）

### Step 1：環境確認（5 分鐘）
- [ ] `nvidia-smi` 正常輸出
- [ ] `python3 --version` ≥ 3.9
- [ ] 安裝依賴：`pip install pynvml httpx fastapi uvicorn streamlit altair`
- [ ] Clone / rsync 代碼到 `/home/wei/pdd/`

### Step 2：跑診斷（5 分鐘）
- [ ] `python3 cluster/diagnose.py`
- [ ] 記錄 `diagnose_result.json` 關鍵欄位：
  - `gpu_model`: _______________
  - `gpu_count`: ___
  - `gpu_memory_gb`: ___
  - `pynvml_available`: ___
  - `recommended_strategy`: _______________
  - `recommended_sampler_interval_ms`: ___

### Step 3：根據診斷結果決策

**pynvml 可用（85% 機率）→ 主力方案，不需改代碼**
- [ ] 確認 `recommended_sampler_interval_ms` ≤ 10ms → 可以

**pynvml 不可用 → 備案**
- [ ] DCGM 可用？→ 改 `collectors/gpu_power.py` 加 DCGM sampler（100ms）
- [ ] 都不可用？→ nvidia-smi fallback，Demo 時避開 prefill 精度話題

**多卡（gpu_count > 1）**
- [ ] 確認 vLLM 是 tensor parallel 還是多實例？
  - Tensor parallel → energy = Σ 所有 GPU，sampler 已支援（MultiGpuPowerSamplerNVML）
  - 多實例 → 每卡獨立跑 proxy，不需改代碼

### Step 4：確認 vLLM 在跑
- [ ] `curl -s http://localhost:8000/v1/models | python3 -m json.tool`
- [ ] 記錄 model id: _______________

---

## 【週六】校準 + 驗證（目標 60 分鐘）

### Step 1：校準常數（~30 分鐘）
- [ ] `python3 cluster/calibrate.py`
- [ ] ⚠️ 如果 H100 prefill 很快（< 100ms），Round 間 W_PREFILL 差距 > 30%？
  - 是 → 改 `calibrate.py` 的 prompt 為 3200 tokens，或 ROUNDS 改成 5
- [ ] 校準完成，記錄結果：
  - `IDLE_POWER`: ___ W
  - `W_PREFILL`: ___ J/token
  - `W_DECODE`: ___ J/token
  - `ratio` (W_D/W_P): ___x  ← **這是 Demo 論點②的數字**
- [ ] 手動更新 `attribution_engine.py` 三個常數

### 🔴 關鍵驗證：H100 prefill/decode 比例

在 calibrate.py 跑完後，立即確認：

- [ ] 計算 W_DECODE / W_PREFILL 比例
- [ ] 比例 > 3x → 假設成立，繼續原定 demo 方向
- [ ] 比例 1-3x → 比例存在但偏低，調整話術（見備案 A）
- [ ] 比例 ≈ 1x → 假設不成立，切換備案 B

**備案 A 話術調整**
> "4060 Ti 上是 8.3x，H100 上是 Xx——不同硬體比例不同，
>  這正是為什麼需要 calibrate，不能用假設。"

**備案 B 話術切換**
核心訊息從「decode 貴 8.3 倍」換成：
> "你知道你的每一個 API call 花了多少電嗎？
>  現在業界沒有工具做到 per-request 成本可視化。
>  PowerDecode 做到了，而且在 64 concurrent 下仍然準確。"

強調：
1. Per-request 成本可視化（業界沒有）
2. Concurrent attribution 正確性（stress test 數據佐證）

### Dashboard AI 分析同步更新

- [ ] 如果比例改變，更新 STATS_CONTEXT 裡的：
      `"Known: decode token costs 8.3x more electricity than prefill token."`
      改成實測數字
- [ ] 如果備案 B，三個 prompt 的核心假設需要重寫
      → 週六確認數字後回報，Claude 當場給新 prompt

### Step 2：跑驗證一二三（~20 分鐘）
- [ ] `python3 cluster/validate_all.py`
- [ ] 驗證一（單 request 閉環）：error < 10%？___ PASS / FAIL
- [ ] 驗證二（並發比例分配）：X error < 15%，Y error < 15%？___ PASS / FAIL
- [ ] 驗證三（能量守恆）：error < 5%？___ PASS / FAIL

**如果 FAIL：**
| 狀況 | 解法 |
|------|------|
| 驗證一 FAIL | 重新確認採樣頻率，檢查 sampler 是否在正確 GPU 上 |
| 驗證二 FAIL | 大概率常數沒校準好，重跑 calibrate.py |
| 驗證三 FAIL | IDLE_POWER 偏差，重測 idle（空閒 20 秒取平均） |

### Step 3：端到端測試（~10 分鐘）
```bash
# Terminal 1
python3 proxy.py

# Terminal 2（重複三次，間隔 2s）
curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```
- [ ] DB 有 3 筆 record，`energy_joules > 0`，`cost > 0`
- [ ] `/health` 回傳 `{"status": "ok", "vllm": "ok"}`
- [ ] `/stats/recent` 回傳正確資料
- [ ] 第 2、3 筆 energy 數值穩定（第 1 筆可能偏低，屬正常 warm-up 效應）

### Step 5：備份 Demo 數據到本機
- [ ] 在 Fluidstack 上：`bash cluster/backup_remote.sh`
- [ ] 確認輸出 ≥ 30 筆、有 anomaly 數據
- [ ] 在本機執行 scp 指令（腳本會印出）
- [ ] 在本機：`bash cluster/restore_local.sh ~/pdd_demo_backup.db`
- [ ] 本機 `streamlit run dashboard.py` 確認資料正確顯示

### Step 6：記錄週六數據（週日用）
```
硬體：_____________ x ___
IDLE_POWER = ___ W
W_PREFILL  = ___ W/tok
W_DECODE   = ___ W/tok
Ratio      = ___x

驗證一 error: ___%  PASS/FAIL
驗證二 X/Y:   ___% / ___%  PASS/FAIL
驗證三 error: ___%  PASS/FAIL
```

---

## 【週日 11:30am】Hacking 開始（目標 30 分鐘上線）

- [ ] 確認 vLLM 在跑：`curl -s http://localhost:8000/v1/models`
- [ ] 填入週六校準常數：`vim attribution_engine.py`
- [ ] 清掉舊 DB：`rm -f data/powerdecode.db`
- [ ] 啟動 proxy：`python3 proxy.py &`
- [ ] 送 3 個 warm-up requests（消除 GPU 冷啟動效應）
- [ ] 快速驗證：`python3 cluster/validate_all.py`（預期全 PASS）
- [ ] 啟動 Dashboard：`streamlit run dashboard.py --server.port 8501 --server.headless true &`
- [ ] 塞 Demo 資料：送幾個不同長度的 request，讓 Dashboard 有資料
- [ ] 瀏覽器打開 `http://localhost:8501` 確認 Dashboard 正常顯示

### Demo 就緒最終確認
- [ ] proxy 在跑（port 8001）
- [ ] dashboard 在跑（port 8501）
- [ ] DB 有 10+ 筆 request
- [ ] Dashboard Overview 頁面有數據、有 cost_usd 數字
- [ ] `/health` 回傳 ok

---

## 【週日 5pm】Demo 論點備忘

1. **定位**：「Lighthouse 告訴你叢集健不健康，PowerDecode 告訴你錢花在哪」
2. **核心數字**：「Decode token 電耗是 prefill 的 ___x（填 H100 實測）」
3. **方法論**：「功耗非線性 → vLLM continuous batching 固定功耗池 → 加權分配是最公平的計費方式」
4. **護城河**：「這份數據是業界第一份推理成本基準報告的原始材料」

---

## 緊急應對

| 狀況 | 應對 |
|------|------|
| pynvml 裝不了 | `pip install nvidia-ml-py`，都不行用 nvidia-smi fallback |
| vLLM OOM | `gpu_memory_utilization=0.70`，`max_num_seqs=16` |
| 採樣太慢（> 20ms） | 調高 interval，Demo 時避開 prefill 精度話題 |
| 驗證全 FAIL | 重新校準 IDLE_POWER，最常見錯誤來源 |
| 多卡不知道怎麼改 | 先用單卡跑 Demo，多卡歸因當 future work |
| Dashboard 打不開 | `pip install --upgrade streamlit`，或直接用 `/stats/recent` API 展示 |
| DB 被鎖 | `rm data/powerdecode.db`，重啟 proxy |
| W_PREFILL 三輪差距 > 30% | H100 prefill 太快，改用 3200 token prompt 重新校準 |
