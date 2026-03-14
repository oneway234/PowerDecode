# PowerDecode Demo 準備策略

> 更新：2026-03-13 週五晚
> 硬體：2x B200 節點（16 GPUs），Slurm，user-010

---

## 一、競爭態勢

### 你的位置
- 11 個 Slurm 頂級項目之一（user-010）
- 唯一做 per-request 電費拆分的項目
- 技術完整度最高：實測校準 + 三個驗證 + dashboard + AI 三維分析

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
- Dylan Patel（SemiAnalysis）— GPU 算力商業邏輯專家
- Gary Wu（Fluidstack）— 直接利益相關者
- Thomas Raoux（OpenAI）
- Mark Saroufim（GPU Mode）
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

## 三、Demo 核心論點（5 分鐘）

### 論點順序
1. **定位**（30秒）
   > 「Lighthouse 告訴你叢集健不健康。PowerDecode 告訴你錢花在哪。」

2. **問題**（1分鐘）
   > 「Fluidstack 賣 GPU/hour，看不到 inference 內部發生什麼。
   >  OpenAI 在 API 層猜到了 output token 更貴。我們在 GPU 硬體層量到了。」

3. **核心發現**（1分鐘）
   > 「Decode token 耗電是 prefill 的 Xx（填 B200 實測數字）。
   >  這不是理論值，是 B200 實測。」
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

## 四、AI 三維分析設計邏輯

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

## 五、關鍵數字（週六填入）

| 數字 | 來源 | 狀態 |
|------|------|------|
| Decode/Prefill ratio = **___x** | B200 calibrate.py | ⬜ 週六填入 |
| IDLE_POWER = **___ W** | B200 calibrate.py | ⬜ 週六填入 |
| V1 error = **___%** | validate_all.py | ⬜ 週六填入 |
| V2 error = **___%** | validate_all.py | ⬜ 週六填入 |
| V3 error = **___%** | validate_all.py | ⬜ 週六填入 |
| 4060 Ti ratio = **8.3x** | 已驗證 | ✅ |

---

## 六、備案

### 備案 A（ratio 1-3x）
話術調整：
> 「4060 Ti 是 8.3x，B200 是 Xx——不同硬體比例不同，
>  這正是為什麼需要在硬體層 calibrate，不能用假設。」

### 備案 B（ratio ≈ 1x）
核心訊息換成：
> 「你知道你的每一個 API call 花了多少電嗎？
>  現在業界沒有工具做到 per-request 成本可視化。
>  PowerDecode 做到了，而且在高並發下仍然準確。」

強調：per-request 可視化 + concurrent attribution 正確性

---

## 七、彩排檢查視角

每個環節問自己兩個問題：

**評審視角**
- 這個數字我為什麼要相信它？
- 這跟 Fluidstack 的生意有什麼關係？

**技術評審視角**
- 8.3x 怎麼量出來的？樣本數多少？
- anomaly 的判定標準是什麼？
- 如果模型不是這個怎麼辦？

---

## 八、勝算評估

**整體：前三有機會**

- 技術完整度最高的項目之一
- 商業敘事最清晰的項目之一
- 唯一做成本歸因的項目

**決定性變數：週六 B200 calibrate 結果**
- ratio > 3x → 勝算 50%+
- ratio 1-3x → 備案 A，勝算 35%
- ratio ≈ 1x → 備案 B，勝算 20%

---

## 九、週六行動清單

```
① SSH 進 B200（Slurm salloc 互動 session）
② tmux 保護所有 process
③ diagnose.py → 確認 pynvml 可用
④ calibrate.py → 拿到 ratio（最重要）
⑤ 根據 ratio 決定 demo 方向
⑥ validate_all.py → 三個驗證 PASS
⑦ stress_test_h100.py（改成 B200 參數）
⑧ seed_demo_data.py
⑨ backup_remote.sh → SCP 回本機
⑩ 本機開 dashboard 確認
⑪ 彩排一次（5 分鐘計時）
```

---

## 十、不要做的事

- 不要在週六加新功能
- 不要改 attribution 核心邏輯
- 不要因為 ratio 跟預期不同就慌——備案已經準備好
- 不要在 demo 時說「我們計劃未來要...」超過一次
