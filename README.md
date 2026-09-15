# Quark W4A16 → SGLang GPTQ：在 RDNA3（gfx1100）上服務 AMD 原版 Quark INT4 模型

**問題**：AMD 在 HuggingFace 放出 `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`，但它在 AMD RDNA3（gfx1100）上**跑不起來**。

SGLang 的 Quark 量化模組只實作了三種 scheme：

    python/sglang/srt/layers/quantization/quark/schemes/
      quark_w4a4_mxfp4.py
      quark_w4a8_mxfp4_moe.py
      quark_w8a8_fp8.py
      ...

純 **W4A16** 沒有實作——`quark/weights.py` 裡直接註明：

    # Scheme didn't allocate the parameter (e.g. W4A16); skip.

**解法**：把 Quark 的 **N-packed `[K, N/8]`** 佈局重新打包成 GPTQ 的 **K-packed `[K/8, N]`**，SGLang 便能以既有的 GPTQ 路徑服務它。**權重未改動任何一個位元。**

---

## 快速開始

    # 1. 取得 AMD 原版模型
    huggingface-cli download amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 \
        --local-dir ./Qwen3.8-27B-Quark-AWQ-INT4-W4A16

    # 2. 轉換（約需 19 GB 磁碟 + 建議 32 GB RAM）
    python3 tools/convert_quark_int4_to_gptq_v3.py \
        ./Qwen3.8-27B-Quark-AWQ-INT4-W4A16 \
        ./Qwen3.8-27B-INT4-GPTQ-v3

    # 3. 以 SGLang 服務（--quantization gptq）
    python3 -m sglang.launch_server \
        --model-path ./Qwen3.8-27B-INT4-GPTQ-v3 \
        --quantization gptq --dtype bfloat16 --kv-cache-dtype bf16 \
        --served-model-name qwen3.8-27b --port 8080

來源目錄與目的目錄亦可改用環境變數 `QUARK_SRC` / `GPTQ_DST` 指定。

---

## 轉換的關鍵：Quark 的 nibble 重排

這是整個工作的技術核心，也是我們耗時最久的地方。

Quark 在 `pack_method="reorder"` 之下，每個 int32 內的 8 個 4-bit 值**並非順序排列**，而是按：

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

重排。**忽略這一步，整個模型將輸出亂碼**——不是數值偏差，是完全不可讀的文字。

我們最初的版本正是遺漏了這一步，以數值掃描猜測 nibble 語義數小時之久。最終閱讀官方 `quark/torch/utils/pack.py` 的 `Pack_4_bits` 之後即得解。

**教訓：凡涉及外部格式，先讀官方實作，不可依靠試探。**

轉換器實作了完整語義：

| 步驟 | 內容 |
|---|---|
| unpack | nibble `i` ← 真值 index `c*8 + [0,2,4,6,1,3,5,7][i]` |
| 還原序 | `unpacked[:, ORDER]`，`ORDER = [0,4,1,5,2,6,3,7]` |
| 數值型別 | int4 為**有號二補數**（−8..7） |
| 尺度 | per-group 有轉置（pack 前先 `weight.T`），scale 形狀 `(K/gs, N)` |
| 對齊 GPTQ kernel | kernel 語義為 `w = (q - (qzeros_nibble+1)) * scale`；令 `q = signed4 + 8`、`qzeros nibble = 7` ⇒ `w = signed4 * scale` ✓ |

---

## 無損驗證

| 檢驗 | 結果 |
|---|---|
| 逐元素反量化比對（抽驗 5 層，含 K=5120／N=17408 與 N=48 的極瘦層） | **max 絕對差 = 0.000000e+00** |
| 全層覆蓋 | **496 個量化層 ＋ 703 個普通層**，全數通過 |
| 對 BF16 原模型的忠實度（96 個 `in_proj` 模組） | cos **最小 0.9871、中位 0.9905** |

第三項顯示 AWQ 量化本身正常——dequant 後與 BF16 原模型的餘弦相似度都在 0.987 以上。

驗證方法（可自行複核）：兩邊各自按自己的語義反量化，逐元素相減。

---

## 我們量到的（GIGABYTE Radeon™ PRO W7800 AI TOP 48G／單卡／70 CU／gfx1100／ROCm 7.2.4）

| 項 | 值 | 條件 |
|---|---|---|
| decode（DSH 真實 agent 路徑） | **62.11 t/s** | 上下文約 20K，n=11 |
| decode（短提示單流） | **85.05 t/s** | 同一台機器、同一配置 |
| Wikitext word_perplexity | **9.7297** | 對 AMD 官方卡 8.8250（+10.25%） |
| GSM8K 5-shot non-thinking | **83.70% / 84.12%** | flexible / strict，兩次全量平均，greedy |

⚠️ **兩點必須提醒：**

1. **我們的 greedy 解碼不確定。** 同機同配置、同樣 150 題連跑兩次，輸出只有 71.3% 相同。故單次跑分 = 真分數 + 噪音（約 ±1 pt）。詳見 `docs/NONDETERMINISM-GREEDY-20260915.md`。
2. **解碼制度未對齊。** 我們用 greedy，AMD 官方卡用 `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`。

**與 AMD 公布數字的差距（PPL +10.25%、GSM8K −7.81 pt）我們尚未定位機制。** 已排除六項嫌疑：權重量化（逐位元相同）、LM head、量測儀器、資料集、prompt 截斷、我們自己的 K-split 優化。詳見 `docs/SCOREBOARD-20260915.md`。

---

## 倉庫內容

| 路徑 | 內容 |
|---|---|
| `tools/convert_quark_int4_to_gptq_v3.py` | 轉換器本體（本倉唯一可執行檔） |
| `docs/SCOREBOARD-20260915.md` | 全部實測分數的單一記錄點 |
| `docs/NONDETERMINISM-GREEDY-20260915.md` | greedy 非確定性的定量報告 |
| `docs/COMMUNITY-POST-20260915.md` | 對應的論壇帖文（含完整致謝、失敗記錄與技術附錄） |

---

## 發佈政策

- 本倉只放**原始碼**與**可獨立驗證的報告**。
- **生成物不入版控**：由 Markdown 衍生的 PDF／TXT 等一律不提交，需要時自行以任一 Markdown 渲染器產生。
- 每一項數字都附量測條件與樣本數；任何無法復現的數字不列入。

---

## 已知限制

- **只處理單一 `model.safetensors`** 的模型目錄（本轉換器針對 AMD 原版模型的佈局）。
- **不含 kernel 改動。** 本倉所述「我們的做法」限於轉換層；我們的 SGLang fork 另有 kernel 層改動（相對上游 18 commit／42 檔），尚未在此發佈。帖文第六節已如實交代。
- 需要約 19 GB 磁碟空間做中轉。

---

## 授權與出處

- 本轉換器為我們自行撰寫，以 **Apache-2.0** 釋出。
- 它依賴的 Quark 打包語義來自 [AMD Quark](https://github.com/amd/Quark)（Apache-2.0）。
- 目標格式對齊 [vLLM](https://github.com/vllm-project/vllm) 的 GPTQ kernel 語義（Apache-2.0）。
- 模型本身來自 AMD（`amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`）。
