# PowerDecode — Claude Code 項目指南

> LLM inference 成本透明化工具：私有層看清每個 request 真實成本，公共層建立業界首個匿名 inference 成本 benchmark。
> 定位：Fluidstack Lighthouse 的成本分析擴充層。

---

## 核心文檔索引

| 文檔 | 路徑 | 說明 |
|------|------|------|
| 設計文檔 | `PowerDecode_DesignDoc.md` | **最重要** — 產品定位/架構/Attribution 方法/實驗設計/Demo 規劃 |

---

## 產品概述

PowerDecode 串聯三層從未被關聯的數據：

```
電力層   →  GPU 功耗（nvidia-smi / DCGM）
推理層   →  tokens, latency（vLLM /metrics）
業務層   →  request ID, endpoint, model version
```

透過統一 request ID 串聯三層，計算每個 request 的真實成本，並拆分 Prefill / Decode 成本。

---

## 技術選型

| 元件 | 選擇 | 原因 |
|---|---|---|
| Inference Server | vLLM | 有完整 /metrics endpoint，prefill/decode 分開提供 |
| 功耗採樣 | nvidia-smi | 消費級卡和無 root 環境都能用，DCGM 為進階選項 |
| 後端 | Python FastAPI | 串聯三層數據的中間層 |
| Dashboard | Streamlit | Python 全搞定，最快出 Demo |
| AI 分析 | Claude API | 異常偵測 + 自然語言解釋 |

---

## 項目結構

```
pdd/
├── CLAUDE.md                    # 本文件 — Claude Code 項目指南
├── PowerDecode_DesignDoc.md     # 設計文檔 v0.2
├── collectors/                  # 數據採集層
│   ├── gpu_power.py             # nvidia-smi 功耗採樣（100ms 間隔）
│   ├── vllm_metrics.py          # vLLM /metrics 抓取
│   └── request_interceptor.py   # 業務層 request ID 串聯
├── engine/                      # 核心計算
│   └── attribution.py           # 成本歸因引擎（時間切片 + 加權 token 分配）
├── api/                         # FastAPI 後端
│   └── main.py                  # API 入口
├── dashboard/                   # Streamlit Dashboard
│   └── app.py                   # 三頁面 Dashboard（總覽/明細/趨勢）
├── analysis/                    # AI 分析
│   └── claude_analyzer.py       # Claude API 異常偵測 + 自然語言解釋
├── experiments/                 # 實驗數據與腳本
│   └── exp1_prefill_decode/     # 實驗一：Prefill vs Decode 功耗可測性
├── tests/                       # 測試
└── requirements.txt
```

> ⚠️ 以上為規劃結構，隨開發推進建立。

---

## 開發要點

### 開發環境

- **GPU**: NVIDIA RTX 4060 Ti
- **模型**: Llama 3.2 3B（或類似大小開源模型）
- **Python**: 3.10+（使用 type hints `X | None` 語法）
- **套件管理**: pip + requirements.txt

### Attribution 核心邏輯

```
加權 tokens = prefill_tokens × W_prefill + decode_tokens × W_decode
```

- W_prefill / W_decode 從實驗一校正得出，**不是固定值**
- 同模型 + 同 GPU → 可用平均值
- 換模型或換 GPU → 必須重新校正
- **⚠️ 目前 W 值待定**，等實驗一完成後填入

### nvidia-smi 採樣

- 目標間隔：100ms
- 記錄：功耗時間序列、平均瓦數、峰值瓦數
- 如果 100ms 精度不足，需調整時間片大小

### vLLM Metrics

- 從 `/metrics` endpoint 抓取
- 需要的數據：prefill token 數、decode token 數、各自 latency
- 計算：watts per prefill token、watts per decode token

---

## 開發階段

### 階段一：24 小時版（保底 Demo）
- [ ] Mock GPU telemetry 數據
- [ ] vLLM 啟動，抓 /metrics
- [ ] 基本 cost-per-token 計算
- [ ] Streamlit dashboard 顯示結果
- [ ] Claude API 一句話分析

### 階段二：36 小時版（目標）
- [ ] nvidia-smi 真實功耗接入
- [ ] Prefill / Decode 成本拆分
- [ ] Request 歷史比較
- [ ] 異常告警

### 階段三：48 小時版（加分）
- [ ] SLURM job log 整合
- [ ] 效率回歸偵測
- [ ] Benchmark 數據上傳架構

---

## 實驗追蹤

### 實驗一：Prefill vs Decode 功耗可測性驗證

**狀態**: ⏳ 待執行

**判斷標準**:
- A（純 Prefill）和 B（純 Decode）功耗差異 >15% → 加權方法可行
- 差異 <15% → 改用純 time slice，放棄加權

**結果待填入**:
- W_prefill = ?
- W_decode = ?
- nvidia-smi 100ms 採樣精度是否足夠 = ?

---

## 開放問題

| 問題 | 狀態 |
|---|---|
| W_prefill / W_decode 實際比值 | ⏳ 等實驗一 |
| nvidia-smi 100ms 採樣精度 | ⏳ 等實驗一 |
| 多 request 並發 attribution 準確性 | 🔲 未討論 |
| Benchmark 數據匿名化機制 | 🔲 未討論 |

---

## 任務完成後文檔更新清單

| # | 文件 | 更新條件 |
|---|------|---------|
| 1 | `CLAUDE.md` | 項目結構變化、技術選型變化、實驗結果、開發階段進度 |
| 2 | `PowerDecode_DesignDoc.md` | 實驗結果填入、Attribution 方法更新、開放問題解決 |

---

## 關鍵決策速查

| 決策 | 選擇 |
|------|------|
| 成本歸因方法 | 細粒度時間切片 + 加權 token 比例分配 |
| 功耗採樣工具 | nvidia-smi（消費級卡兼容），DCGM 為進階選項 |
| Dashboard 框架 | Streamlit（不用 React，Python 全搞定） |
| AI 分析 | Claude API（異常偵測 + 自然語言解釋） |
| 校正原則 | 同模型同 GPU 用平均值，換任何變數重新校正 |
| Demo 定位 | 看起來像 Lighthouse 擴充功能的 Web Dashboard |
