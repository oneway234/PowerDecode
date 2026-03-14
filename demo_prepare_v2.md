# PowerDecode Demo 準備策略 v2

> 更新：2026-03-13 週五晚
> 硬體：2x B200 節點（16 GPUs），Slurm，user-010
> v2 新增：2 分鐘節奏、評審導向策略、實戰演示順序

---

## 一、競爭態勢

### 你的位置
- 11 個 Slurm 頂級項目之一（user-010）
- 唯一做 per-request 電費拆分的項目
- 技術完整度最高：實測校準 + 五個驗證 + dashboard + AI 三維分析

### 主要對手
| 對手 | 方向 | 威脅程度 |
|------|------|---------|
| user-004 Lucca | GPU cluster 健康監控 | 中（監控方向，但不是成本層） |
| user-009 Andi H | Slurm monitoring agent | 低（Slurm 層，不是 inference 層） |
| user-007 Tristan（Meta） | Speculative decoding | 中（技術深度強，但方向不同） |

### 核心差異化
> Lighthouse 告訴你叢集健不健康。
> PowerDecode 告訴你錢花在哪。
> 這是 Fluidstack 現在缺的那一頁。

---

## 二、評審視角

### 評審陣容
- **Dylan Patel**（SemiAnalysis）— inference economics、tokens per megawatt、做 InferenceX v2
- **Gary Wu**（Fluidstack CEO）— operator、Oxford Economics 出身、在意落地和成本結構
- **Thomas Raoux**（OpenAI）— Triton compiler、GPU kernel 效率、memory bandwidth vs compute bound
- **Mark Saroufim**（GPU Mode）— GPU programming 普及化、benchmark、end-to-end system optimization
- Horace He（Thinking Machines）

### 打到評審痛點的話術
> 「API 層（OpenAI/Anthropic）已經發現 output token 比 input token 貴，
>  開始分開收費。但 Fluidstack 賣裸 GPU/hour，完全看不到 prefill/decode 的分裂。
>  PowerDecode 是第一個在 GPU 硬體層量到這個不對稱的工具。」

### Fluidstack 的商業邏輯
- 賣 H100 $2.10/GPU/hour，B200 更貴
- 算力供不應求，目標不是降價，是**把有限算力賣到最高價值**
- Inference 佔 AI workload 比例從 20% 走向 80%，這個問題只會越來越大

---

## 三、Demo 節奏（2 分鐘版本）（新增 v2）

> 2 分鐘，4 個 beat，每 30 秒打一個評審痛點。

| 時間 | Beat | 畫面 | 核心句 | 打誰 |
|------|------|------|--------|------|
| 0–20s | 定位 | Dashboard 已開，停在 Request Detail | "PowerDecode measures the true electricity cost of every single inference request — from GPU watts to tokens." | 全場 |
| 20–60s | 核心發現 | Request Detail donut chart → 指向 decode 佔比 | "Decode tokens consume significantly more GPU energy than prefill tokens. On this GPU we measured **___x**. This is not a model — it's a B200 measurement." | Dylan |
| 60–90s | 商業含義 | 切到 Pricing tab（AI Insight） | "GPU clouds charge per GPU-hour, but workloads have very different true electricity costs. Your decode-heavy customers are underpaying. Here's how much." （讓評審自己讀 conclusion，不要你念） | Gary |
| 90–120s | 系統價值 | 切到 Overview（全景） | "You can't optimize what you can't measure. PowerDecode is the measurement layer for inference cost optimization." | Mark |

**關鍵原則**：
- **不要從 Overview 開始** — 會被誤認為又一個 monitoring dashboard
- 從 Request Detail 開始 — 立刻展示 per-request granularity，這是差異化
- 讓數字說話，不念數字

---

## 四、Demo 畫面順序（新增 v2）

> 順序決定評審的第一印象。錯誤的順序 = 被歸類為 monitoring tool。

```
1. Request Detail（donut chart — decode 73% 電費）
   → 這是新東西，沒人做過

2. AI Insight — Pricing tab
   → 讓評審自己讀 revenue leak 數字

3. Overview（全景 + anomaly 標記）
   → 收尾，展示系統完整性

4. （如果有時間）Cost Trend
   → 展示不同 request 的成本差異
```

**絕對不要**：
- 從 Overview 開始（看起來像 Grafana）
- 花時間解釋 tech stack
- 展示 terminal / code

---

## 五、評審視線策略（新增 v2）

> 眼神接觸 = 讓評審覺得「這是為我做的」。

| 說到這句話 | 看向 | 原因 |
|-----------|------|------|
| "Decode tokens consume ___x more energy than prefill" | **Dylan** | 他做 TCO 分析，tokens per megawatt 是他的語言 |
| "Your customers have very different true costs" | **Gary** | 他的錢、他的客戶 |
| "Memory-bound vs compute-bound explains the asymmetry" | **Thomas** | 他做 compiler，這是他的領域 |
| "You can't optimize what you can't measure" | **Mark** | 他做 benchmark，這是他的信念 |
| "This is the missing page after Lighthouse" | **Gary** | 直接連結到他的產品線 |
| "Not a model — a B200 measurement" | **Dylan** | 他做 InferenceX，他在乎的是實測不是模型 |

---

## 六、評審心理節奏（新增 v2）

> Demo 的每 30 秒在回答不同評審腦中的問題。

| 時段 | 評審腦中的問題 | 你在回答 |
|------|--------------|---------|
| 0–30s | "這是什麼？又一個 dashboard？" | 不是 monitoring。是 per-request 成本量測。 |
| 30–60s | "數字可信嗎？怎麼量的？" | B200 實測，不是模型。Decode ___x prefill。 |
| 60–90s | "跟我有什麼關係？" | 你的客戶在少付錢。量化了多少。 |
| 90–120s | "這能做什麼？" | 這是 inference optimization 的量測基礎設施。 |

**每個評審的核心關注**：

| 評審 | 關心 | 你給他的答案 |
|------|------|-------------|
| Dylan | inference economics | "InferenceX compares hardware. PowerDecode compares requests on the same hardware." |
| Gary | customer profitability | "Which customer segment is actually profitable at flat GPU/hour pricing?" |
| Thomas | 技術合理性 | "Prefill is compute-bound, decode is memory-bound. That's why the power profiles differ." |
| Mark | optimization measurement | "PowerDecode is the measurement layer that makes end-to-end optimization possible." |

---

## 七、Demo 話術簡化原則（新增 v2）

> 評審不關心 tech stack。他們關心 insight。

**不要說**：
- "We use FastAPI as our proxy layer"
- "NVML samples at 10ms intervals"
- "SQLite stores the attribution results"
- "Trapezoidal integration minus idle power"

**只需說**：
- "We measure GPU watts per request"
- "Real hardware measurement, not a model"
- "Calibrated on this GPU"

**如果被問技術細節**（Thomas 可能會問）：
- 簡短回答："pynvml at 10ms, trapezoidal integration, weighted token allocation"
- 然後馬上拉回商業："The method is validated — three tests pass within 5% error"

---

## 八、Demo 核心論點（5 分鐘版）

### 論點順序
1. **定位**（30秒）
   > 「Lighthouse 告訴你叢集健不健康。PowerDecode 告訴你錢花在哪。」

2. **問題**（1分鐘）
   > 「Fluidstack 賣 GPU/hour，看不到 inference 內部發生什麼。
   >  OpenAI 在 API 層猜到了 output token 更貴。我們在 GPU 硬體層量到了。」

3. **核心發現**（1分鐘）
   > 「Decode tokens consume significantly more GPU energy than prefill tokens.
   >  On this GPU we measured Xx. This is a B200 measurement, not a model.」
   >
   > 打開 Request Detail → donut chart：
   > 「這一筆 request，73% 的電費來自 decode。
   >  按 token 數量平均收費，decode-heavy 用戶少付了錢。」

4. **三維分析**（1分鐘）
   > 點「▶ Three-Perspective Analysis」
   > 讓評審自己讀定價視角的 conclusion，不要你念。

5. **Slider + 收尾**（30秒）
   > 拖動電價 slider：「如果你的 datacenter 在德國，成本跟著變。」
   >
   > 收尾：
   > 「Inference 佔 AI workload 的比例正在從 20% 走向 80%。
   >  Fluidstack 現在沒有工具看到這塊的成本結構。
   >  PowerDecode 是第一步。」

---

## 九、AI 三維分析設計邏輯

### 三個視角
| 視角 | 目標受眾 | 核心訊息 |
|------|---------|---------|
| ☁️ 定價視角 | Fluidstack | 你正在少收 decode-heavy 用戶的錢，量化缺口 |
| 🔍 運營視角 | 叢集運營商 | anomaly pattern，哪 X% request 消耗 Y% 能耗 |
| 💰 用戶視角 | 開發者 | flat-rate 下你是補貼者還是被補貼者 |

### 關鍵背景（已寫入 prompt）
- Fluidstack 賣裸 GPU/hour，看不到 prefill/decode 分裂
- OpenAI/Anthropic 已在 API 層分開收費（output 貴 3-5x）
- PowerDecode 是第一個在硬體層量到這個不對稱的工具

---

## 十、Demo 成功條件（新增 v2）

### 30 秒內評審必須理解：
- [ ] PowerDecode measures per-request inference electricity cost
- [ ] Decode vs Prefill have different power profiles
- [ ] Flat GPU/hour pricing hides this asymmetry

### 2 分鐘內評審必須看到：
- [ ] Request energy breakdown（donut chart）
- [ ] Pricing impact（AI Insight revenue leak 數字）
- [ ] Live system running（不是 slides）

### Demo 結束時評審必須記住：
- [ ] "Decode 比 prefill 貴 ___x"（一個數字）
- [ ] "Lighthouse 看健康，PowerDecode 看成本"（一句定位）
- [ ] "這是 Fluidstack 缺的那一頁"（一個 hook）

---

## 十一、關鍵數字（週六填入）

| 數字 | 來源 | 狀態 |
|------|------|------|
| Decode/Prefill ratio = **___x** | B200 calibrate.py | ⬜ 週六填入 |
| IDLE_POWER = **___ W** | B200 calibrate.py | ⬜ 週六填入 |
| V1 error = **___%** | validate_all.py | ⬜ 週六填入 |
| V2 error = **___%** | validate_all.py | ⬜ 週六填入 |
| V3 error = **___%** | validate_all.py | ⬜ 週六填入 |
| 4060 Ti ratio = **8.3x** | 已驗證 | ✅ |

---

## 十二、備案

### 備案 A（ratio 1-3x）
話術調整：
> "On a 4060 Ti we measured 8.3x. On this B200 we measured Xx.
>  Different GPU architectures yield different ratios —
>  that's exactly why you need hardware-level calibration, not assumptions."

### 備案 B（ratio ≈ 1x）
核心訊息換成：
> 「你知道你的每一個 API call 花了多少電嗎？
>  現在業界沒有工具做到 per-request 成本可視化。
>  PowerDecode 做到了，而且在高並發下仍然準確。」

強調：per-request 可視化 + concurrent attribution 正確性

---

## 十三、彩排檢查視角

每個環節問自己兩個問題：

**評審視角**
- 這個數字我為什麼要相信它？
- 這跟 Fluidstack 的生意有什麼關係？

**技術評審視角**
- ratio 怎麼量出來的？樣本數多少？
- anomaly 的判定標準是什麼？
- 如果模型不是這個怎麼辦？

---

## 十四、週六彩排策略（新增 v2）

> 至少彩排三次。超時 = 刪句子，不加句子。

### 2 分鐘版（主力）

| 段落 | 目標時間 | 內容 |
|------|---------|------|
| 定位 | 15s | 一句話 + Request Detail 已開 |
| 核心發現 | 35s | Donut chart + ratio 數字 |
| 商業含義 | 30s | Pricing tab + revenue leak |
| 系統價值 | 25s | Overview 全景 + 收尾句 |
| **合計** | **105s** | 留 15s buffer |

### 5 分鐘版（如果時間允許）

用八、Demo 核心論點的完整版本。加入 slider demo 和更多 AI Insight tabs。

### 彩排方法
1. 計時器開著講一遍
2. 記下超時的段落
3. 刪掉最弱的句子（不是加快語速）
4. 重複直到 < 105s

---

## 十五、勝算評估

**整體：前三有機會**

- 技術完整度最高的項目之一
- 商業敘事最清晰的項目之一
- 唯一做成本歸因的項目

**決定性變數：週六 B200 calibrate 結果**
- ratio > 3x → 勝算 50%+
- ratio 1-3x → 備案 A，勝算 35%
- ratio ≈ 1x → 備案 B，勝算 20%

---

## 十六、週六行動清單

```
① SSH 進 B200（Slurm salloc 互動 session）
② tmux 保護所有 process
③ diagnose.py → 確認 pynvml 可用
④ calibrate.py → 拿到 ratio（最重要）
⑤ 根據 ratio 決定 demo 方向
⑥ validate_all.py → 五個驗證
⑦ stress_test_h100.py（改成 B200 參數）
⑧ seed_demo_data.py
⑨ backup_remote.sh → SCP 回本機
⑩ 本機開 dashboard 確認
⑪ 彩排三次（2 分鐘計時）
⑫ 確認 demo 畫面順序：Detail → Pricing → Overview
```

---

## 十七、不要做的事

- 不要在週六加新功能
- 不要改 attribution 核心邏輯
- 不要因為 ratio 跟預期不同就慌——備案已經準備好
- 不要在 demo 時說「我們計劃未來要...」超過一次
- 不要從 Overview 開始 demo
- 不要講 tech stack（FastAPI、NVML、SQLite）
- 不要念 AI Insight 的文字——讓評審自己讀
