# PowerDecode — Design Doc v0.2
> 狀態：草稿，實驗一完成後更新權重設計
> 作者：Wei Wang
> 最後更新：2026-03-12

---

## 一、一句話定義

> 讓 LLM inference 成本從黑盒變透明：私有層看清每個 request 的真實成本，公共層建立業界第一個匿名 inference 成本 benchmark。

---

## 二、目標用戶與產品定位

### 用戶路徑

PowerDecode 的定位不是獨立工具，而是 **Fluidstack Lighthouse 缺少的那一層**。

```
現在 Lighthouse 能做的：
  ✅ Cluster 健康監控
  ✅ 自動重啟失敗的 job
  ✅ 硬體異常偵測

Lighthouse 看不到的：
  ❌ 每個 request 花了多少錢
  ❌ Prefill vs Decode 成本拆分
  ❌ 哪個 endpoint / 模型版本最燒錢
```

PowerDecode 填補這個空白。

### 用戶角色

| 角色 | 他們要的 | PowerDecode 給他們什麼 |
|---|---|---|
| ML Engineer | 知道新模型上線後成本有沒有變貴 | Per-request 成本歷史比較 |
| Infra / DevOps | 知道哪個 job 在異常消耗資源 | 即時異常告警 |
| AI 公司 CTO | 控制 inference 總成本 | Endpoint 成本排行 |
| Fluidstack（買單方） | 給客戶更透明的帳單，作為差異化競爭優勢 | Request-level 成本明細 |

### Demo 當天的核心說法

> 「Fluidstack 的 Lighthouse 告訴你 cluster 健不健康。PowerDecode 告訴你錢花在哪裡。這是 Lighthouse 的下一頁。」

---

## 三、要解決的問題

### 根本問題
LLM inference 的真實成本對所有人都是黑盒。

現在系統能告訴你：
- GPU 使用率
- Token 數量
- 月帳單總額

但不能告訴你：
- 每個 request 花了多少錢
- Prefill 和 Decode 各佔多少成本
- 從電費到 token 的完整成本鏈條

---

### 私有層問題

**問題一：客戶端單體成本不透明**

現況：
- 公司只看到整體 GPU bill
- 或 cluster utilization

看不到：
- 每個 request 的成本
- 每個 endpoint 的成本
- 每個模型版本的成本

影響：
- 成本優化沒有方向
- Feature ROI 無法計算
- Inference cost debugging 極難

**問題二：Prefill / Decode 成本從未被拆開**

LLM inference 有兩個本質不同的階段：

| | Prefill | Decode |
|---|---|---|
| 特性 | Compute bound | Memory bound |
| 時間 | 短 | 長 |
| 每 token 耗電 | 高（待實驗驗證） | 低（待實驗驗證） |

現況：所有工具把兩者混在一起計算。

影響：
- 無法知道 KV cache 優化的實際效果
- Batching 參數調整的成本影響不可見
- 模型版本更新的成本變化無從追蹤

**技術難點：三層數據從未被串聯**

這不是用戶痛點，但是解決問題一的前提：

```
電力層   →  GPU 功耗（nvidia-smi / DCGM）   →  Infra 管
推理層   →  tokens, latency（vLLM metrics）  →  ML 管
業務層   →  request, endpoint, model version →  產品管
```

三層數據沒有統一的 request ID 串聯，導致問題一無法被解決。

---

### 公共層問題

**問題三：業界沒有 inference 成本 benchmark**

目前沒有人能回答：
- H100 + Llama 3 8B，正常的 cost/token 是多少？
- 同樣的模型，A100 vs H100 的成本差異是多少？
- Batch size 從 8 改到 32，成本怎麼變？

每家公司各自摸索，沒有任何業界基準可以對照。

這個缺失導致：
- 模型選擇只看 benchmark 準確率，不看成本
- 參數調優（batch size, context length）沒有成本反饋
- 模型版本更新後，成本影響要等月底帳單才知道

---

## 三、PowerDecode 的解法

### 核心架構

```
[nvidia-smi / DCGM]     → 電力層：每 100ms 採樣一次 GPU 功耗
[vLLM /metrics]         → 推理層：每個 request 的 token 數、latency、prefill/decode 分拆
[Request Interceptor]   → 業務層：request ID、endpoint、model version

          ↓ 統一 request ID 串聯三層

[Attribution Engine]    → 計算每個 request 的真實成本
[Streamlit Dashboard]   → 可視化
[Claude API]            → 異常偵測 + 自然語言解釋
[Benchmark Layer]       → 匿名貢獻數據，建立公共基準（未來）
```

### Sidecar 部署架構

PowerDecode 必須作為 sidecar process 部署在與 vLLM 同一台機器上（需要本地讀取 GPU 功耗）：

```
同一台機器：
vLLM（port 8000）           ← inference server
PowerDecode（port 8001）    ← 成本計算引擎
  ├── 讀 nvidia-smi / NVML  ← 電力層數據
  ├── 讀 vLLM /metrics      ← 推理層數據
  └── Attribution Engine     ← 成本歸因計算
Streamlit Dashboard（port 8501） ← 可視化
```

---

### Attribution 方法

**基本邏輯：細粒度時間切片 + 加權 token 比例分配**

步驟：
1. 把時間切成細粒度單位（目標 100ms，依實驗結果調整）
2. 每個時間片內，記錄有哪些 request 在運行
3. 按每個 request 在這個時間片內的**加權 token 數**比例分配電費

**加權公式（待實驗一驗證）：**

```
加權 tokens = prefill_tokens × W_prefill + decode_tokens × W_decode

W_prefill 和 W_decode 從校正實驗得出，不是拍腦袋的固定值
```

**Idle Power 扣除：**

GPU 在閒置狀態仍有固定功耗（idle power），這部分不應分攤給任何 request：

```
可分攤功耗 = 總功耗 - GPU idle power
每個 request 分到的功耗 = 可分攤功耗 × (該 request 加權 tokens / 時間片內所有 request 加權 tokens 總和)
```

idle power 的具體數值需要透過實驗零測量。

**校正原則（第一性原理）：**

- 同個模型 + 同個 GPU → 可視為固定，用平均值
- 不同模型 → 必須分別校正
- 不同 GPU → 必須分別校正
- 換了任何一個變數 → 重新跑校正流程

**實驗結果（2026-03-12，4060 Ti + Qwen2.5-3B-Instruct）：**

| 參數 | 數值 |
|---|---|
| W_prefill | 0.0212 W/tok |
| W_decode | 0.1772 W/tok |
| W 比值（decode / prefill） | 8.35x |
| GPU idle power | 21.07W（std 0.017W） |
| Prefill 平均功耗 | 34.7W（5-11 採樣點/輪） |
| Decode 平均功耗 | 106.3W（充足採樣點） |
| 差異百分比 | 206.1%（遠超 15% 門檻） |

⚠️ **已知限制**：Prefill 持續時間僅 56ms，即使 10ms 採樣也只有 5-11 個點，平均值可能被 idle→peak 過渡期拉低。W_prefill 可能被低估。在 hackathon scope 內可接受，Demo 時需誠實說明此限制。

---

## 四、實驗設計

### 實驗零：GPU Idle Power 基準測量

**目的：**
測量 GPU 在無負載狀態下的固定功耗（idle power），作為 Attribution 計算中扣除的基準值。

**方法：**
- 確保 GPU 無任何運算任務
- 只開 nvidia-smi，以 100ms 間隔採樣
- 持續記錄 5 分鐘

**記錄：**
- 平均功耗（watts）
- 最小功耗（watts）
- 標準差（watts）

**判斷標準：**
標準差 < 平均值的 5% → idle power 穩定可用

**結果（2026-03-12）：**
- 平均功耗：21.07W
- 最小功耗：20.97W
- 最大功耗：21.11W
- 標準差：0.017W（佔平均值 0.08%，極度穩定）
- 採樣點數：2995
- 結論：idle power 穩定可用，以 21.07W 作為 attribution 扣除基準

### 實驗一：Prefill vs Decode 功耗可測性驗證

**目的：**
驗證在 nvidia-smi 的採樣精度下，prefill 和 decode 階段的功耗差異是否可以被清楚測量。這是整個 attribution 方法的基礎假設，必須先驗證。

**環境：**
- GPU：NVIDIA RTX 4060 Ti
- 模型：Llama 3.2 3B（或類似大小的開源模型）
- Inference server：vLLM
- 功耗採樣：nvidia-smi，採樣間隔 100ms

**實驗設計：**

測試 A — 純 Prefill 壓力
```
Input：超長 prompt（目標 2000+ tokens）
Output：限制為 1 token
目的：讓 GPU 幾乎只做 prefill，觀察功耗曲線
記錄：整段時間的平均瓦數、峰值瓦數、持續時間
```

測試 B — 純 Decode 壓力
```
Input：極短 prompt（10 tokens 以內）
Output：不限制，讓模型盡量生成（目標 500+ tokens）
目的：讓 GPU 幾乎只做 decode，觀察功耗曲線
記錄：整段時間的平均瓦數、峰值瓦數、持續時間
```

測試 C — 混合（基準對照）
```
Input：中等長度 prompt（200 tokens）
Output：中等長度（200 tokens）
目的：建立混合情境的基準
```

測試 D — 採樣頻率壓力測試
```
目的：找出 NVML 最高穩定採樣頻率
方法：用 pynvml 從 100ms 逐步降到 10ms，記錄每個頻率下的穩定性
記錄：每個頻率的實際採樣間隔、丟失率、功耗讀數穩定性
```

**判斷標準：**

| 結果 | 代表 | 後續行動 |
|---|---|---|
| A 和 B 的功耗有明顯差異（>15%） | 加權方法可行 | 計算 W_prefill / W_decode，更新 Design Doc |
| A 和 B 的功耗差異不明顯（<15%） | 加權方法基礎不成立 | 改用純 time slice，放棄加權，更新 Design Doc |
| prefill 階段採樣點 < 3 | nvidia-smi 採樣精度不足 | 切換到 pynvml 高頻採樣（目標 10-50ms） |

**要記錄的數據：**
```
- nvidia-smi 功耗時間序列（每 100ms 一筆）
- vLLM metrics：prefill token 數、decode token 數、各自的 latency
- 計算：watts per prefill token、watts per decode token
- 每個測試的採樣點數量
- 採樣頻率 vs 測量精度的關係
```

**結果（2026-03-12，pynvml 10ms 採樣）：**

| 測試 | 平均功耗 | Latency | 採樣點/輪 | Tokens (in/out) |
|---|---|---|---|---|
| A（純 Prefill） | 34.7W | 0.056s | 5-11 | 1635 / 1 |
| B（純 Decode） | 106.3W | ~15s | 充足 | ~30 / 600 |
| C（混合） | 96.9W | 4.92s | ~49 | 351 / 200 |

- 環境：4060 Ti 16GB + Qwen/Qwen2.5-3B-Instruct + vLLM 0.17.1
- 採樣器：pynvml 10ms 間隔（GpuPowerSamplerNVML）
- 結論：**加權方法可行**，A vs B 差異 206.1% >> 15% 門檻
- W_prefill = 0.0212 W/tok，W_decode = 0.1772 W/tok

**已知限制**：
- Prefill 的 34.7W 接近 idle（21.07W），只高 13.6W。真實 prefill 峰值可能更高，被短時間採樣的平均效果拉低。
- 測試 D（採樣頻率壓力測試）未執行，pynvml 10ms 已足夠本次實驗需求。

---

## 五、Attribution Engine 驗證設計

### 驗證原則

Attribution Engine 算出來的數字必須可被獨立檢驗。
設計三種驗證方法，按順序執行。

---

### 驗證一：單一 request 封閉驗證（基本正確性）

**條件：** 同一時間只跑一個 request，沒有任何其他 workload。

**邏輯：** 單一 request 時不需要 attribution 分配，所有功耗都屬於這一個 request。Attribution Engine 的結果應該等於直接測量值。

**公式：**
```
直接測量值 = (平均功耗 - idle_power) × 持續時間（秒）
Attribution Engine 結果 = 分配給這個 request 的能量（焦耳）
```

**通過標準：** 兩者誤差 < 10%

**用途：** 驗證 Attribution Engine 的基本計算邏輯是否正確。這是最重要的驗證，必須先通過才能進行後續開發。

---

### 驗證二：已知比例雙 request 驗證（分配邏輯）

**條件：** 同時送兩個 request：
- Request X：純 prefill（長 prompt，max_tokens=1）
- Request Y：純 decode（短 prompt，max_tokens=600）

**邏輯：** 兩個 request 的加權 token 比例是已知的，Attribution Engine 分配的比例應該接近理論值。

**公式：**
```
理論分配比例：
  X 的加權 tokens = prefill_tokens_X × W_prefill
  Y 的加權 tokens = decode_tokens_Y × W_decode
  X 應得比例 = X 加權 tokens / (X + Y 加權 tokens)
```

**通過標準：** 實際分配比例與理論比例誤差 < 15%

**用途：** 驗證多 request 並發時的分配邏輯是否正確。

---

### 驗證三：能量守恆驗證（持續健康檢查）

**條件：** 任意一批 request 跑完後自動執行。

**邏輯：** 所有 request 分到的能量總和，應該等於這段時間 GPU 實際消耗的可分攤能量。

**公式：**
```
可分攤總能量 = Σ (功耗採樣值 - idle_power) × 採樣間隔
Attribution 總能量 = Σ 所有 request 分到的能量
```

**通過標準：** 兩者誤差 < 5%

**用途：** 每次跑完自動驗證，確保沒有能量被遺漏或重複計算。這是 Attribution Engine 的持續性健康檢查。

---

### 驗證四：功耗線性假設驗證（方法正確性）

**條件：** 分兩組測試，每組分別送 1、2、3 個同類型 request 並發。

**測試組 P（純 prefill）：**
- P1：1 個 request，長 prompt（~1600 tokens），max_tokens=1
- P2：同時送 2 個相同 request
- P3：同時送 3 個相同 request

**測試組 D（純 decode）：**
- D1：1 個 request，短 prompt（~30 tokens），max_tokens=600
- D2：同時送 2 個相同 request
- D3：同時送 3 個相同 request

**邏輯：** Attribution Engine 的加權分配公式假設功耗可線性疊加。若 N 個相同 request 並發，功耗應接近 N × 單一 request 功耗。此實驗直接驗證這個假設是否成立。

**公式：**
```
線性係數 = 實際測量功耗 / (N × 單一 request 平均功耗)
```

**通過標準：**
- 線性係數在 0.85 ~ 1.15 之間 → 線性假設成立，Attribution 方法有效
- 線性係數超出此範圍 → 記錄偏差程度，Demo 時說明為已知邊界條件

**用途：** 驗證 Attribution Engine 的核心數學假設。結果不影響 MVP 運行，但影響數字的可信度聲明。若線性假設成立，Demo 可直接主張方法準確；若不成立，誠實標註為「單 request 最準確，並發場景存在 X% 系統誤差」。

**執行時機：** 驗證一二三通過後，有時間再執行。屬於加分項，非 MVP 必要條件。

---

### 驗證執行順序

1. 先跑驗證一，通過後才繼續開發
2. Attribution Engine 完成後跑驗證二
3. 驗證三內建進 Attribution Engine，每次自動執行
4. 驗證四在驗證一二三通過後選做（加分項）

---

### 實驗參數（從實驗結果填入）

| 參數 | 數值 |
|---|---|
| idle_power | 21.07W |
| W_prefill | 0.0212 W/token |
| W_decode | 0.1772 W/token |
| 採樣間隔 | 10ms（pynvml） |

---

## 六、Demo 呈現方式

### 產品定位

Demo 的視覺語言要讓 Gary Wu（Fluidstack CEO）覺得：
> 「這就是我們產品的下一個頁面。」

不是終端機。不是 Jupyter notebook。是一個看起來像 Lighthouse 擴充功能的 Web Dashboard。

---

### Dashboard 設計（Streamlit 實作）

**頁面一：總覽（今日成本）**

```
┌─────────────────────────────────────────────────────┐
│  🔋 PowerDecode  │  Cluster: fluidstack-h100-01     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  今日總成本  $2,847        比昨天 ▲12%              │
│                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │ Prefill 成本          │  │ Decode 成本           │ │
│  │ $1,923  (67%)        │  │ $924   (33%)          │ │
│  └──────────────────────┘  └──────────────────────┘ │
│                                                      │
│  成本最高的 Endpoints                                │
│  ┌────────────────────────────────────────────────┐ │
│  │ /api/summarize   $1,203  (42%)  ⚠️ 異常        │ │
│  │ /api/code        $891    (31%)  ✅ 正常         │ │
│  │ /api/chat        $753    (27%)  ✅ 正常         │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  🤖 Claude 分析                                      │
│  「今日成本比昨天高 12%，主因是 /api/summarize       │
│    的平均 input 長度增加了 340%。                    │
│    建議檢查是否有異常長的 prompt 進入系統。」        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**頁面二：Request 明細**

```
┌─────────────────────────────────────────────────────┐
│  Request 明細                                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  #4521  /api/summarize  llama-3.2-3b  ⚠️            │
│  ─────────────────────────────────────              │
│  總成本：$0.000312                                   │
│  ├── Prefill：$0.000224  (72%)  [2048 tokens]       │
│  └── Decode： $0.000088  (28%)  [128 tokens]        │
│                                                      │
│  GPU 功耗：291W 平均  /  峰值 318W                  │
│  總時間：2.14s  /  Prefill 0.89s  Decode 1.25s      │
│                                                      │
│  比同類 request 貴：+187%                            │
│  原因：input prompt 異常長（正常值 ~600 tokens）     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**頁面三：成本趨勢**

```
  成本/token 趨勢（過去 24 小時）
  
  $0.0004 │     ⚠️
  $0.0003 │  ___/\___
  $0.0002 │ /        \____
  $0.0001 │
          └──────────────────→ 時間
          
  模型版本比較：
  llama-3.2-3b v1  $0.000089/token
  llama-3.2-3b v2  $0.000112/token  ▲ 26%
```

---

### 技術實作（Streamlit）

```python
# 三個頁面用 st.sidebar 切換
# 不需要 React，不需要獨立前端
# Python 全搞定，節省 8-10 小時開發時間
```

---

### Demo 說話方式（5 分鐘版本）

**第 1 分鐘：問題**
> 「你知道你每個月的 GPU 帳單，但你不知道這筆錢是怎麼花掉的。
> Fluidstack 的 Lighthouse 告訴你 cluster 健不健康。
> 但沒有任何工具告訴你錢花在哪裡。」

**第 2 分鐘：解法**
> 「PowerDecode 把三層從來沒有被串聯的數據連起來：
> 電力、推理、業務。統一到一個 request ID。
> 讓你看到每一個 request 從電費到 token 的完整成本。」

**第 3 分鐘：技術亮點**
> 「我們是第一個把 prefill 和 decode 分開計費的工具。
> 這兩個階段的成本結構完全不同，混在一起算是錯的。
> 而且我們的權重是從真實測量得出的，不是假設。」

**第 4 分鐘：Demo**
> 打開 Dashboard，指著 ⚠️ 異常的 request。
> 讓 Claude 的分析說話。
> 「這個 request 比同類貴了 187%，AI 自動找到原因。」

**第 5 分鐘：願景**
> 「這個工具收集的數據，未來可以形成業界第一個
> inference 成本 benchmark。
> 就像 SemiAnalysis 的 ClusterMAX 給 GPU cloud 打分，
> PowerDecode 給 inference 成本建立標準。」

---

## 七、技術選型

| 元件 | 選擇 | 原因 |
|---|---|---|
| Inference Server | vLLM | 有完整的 /metrics endpoint，prefill/decode 分開提供 |
| 功耗採樣 | nvidia-smi + pynvml | nvidia-smi 用於基礎採樣（100ms），pynvml 用於高頻採樣（10-50ms，H100 prefill 太快時的備案） |
| 後端 | Python FastAPI | 串聯三層數據的中間層 |
| Dashboard | Streamlit | 最快，Python 全搞定 |
| AI 分析 | Claude API | 異常偵測 + 自然語言解釋 |

---

## 八、開發範圍

### 24 小時版（保底）
- [ ] Mock GPU telemetry 數據
- [ ] vLLM 啟動，抓 /metrics
- [ ] 基本 cost-per-token 計算
- [ ] Streamlit dashboard 顯示結果
- [ ] Claude API 一句話分析

### 36 小時版（目標）
- [ ] nvidia-smi 真實功耗接入
- [ ] Prefill / Decode 成本拆分
- [ ] Request 歷史比較（比上次貴了嗎？）
- [ ] 異常告警

### 48 小時版（加分）
- [ ] SLURM job log 整合
- [ ] 效率回歸偵測
- [ ] Benchmark 數據上傳架構

---

## 九、待解決的開放問題

| 問題 | 狀態 | 預計解決時間 |
|---|---|---|
| W_prefill / W_decode 的實際比值 | ✅ 0.0212 / 0.1772（比值 8.35x） | 實驗一完成 |
| nvidia-smi 100ms 採樣是否足夠精確 | ✅ 不夠，已切換到 pynvml 10ms | 實驗一完成 |
| 多 request 並發時 attribution 準確性 | 🔲 未討論 | Design Doc v0.2 |
| Benchmark 數據的匿名化機制 | 🔲 未討論 | 48hr 版再討論 |
| GPU idle power 的實際數值 | ✅ 21.07W | 實驗零完成 |
| H100 上 prefill 採樣點是否足夠 | 🔲 未測試（測試D未執行） | 待定 |
| 功耗線性假設是否成立 | 🔲 未執行 | 驗證一二三完成後選做 |

---

*下一步：跑實驗一，根據結果更新第三節的 Attribution 方法*
