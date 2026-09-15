# 低位寬 lm_head shim（INT4 / INT8）

## `rdna_lmhead_int8.py` —— 我們原創

**做法**：經 `PYTHONPATH` 注入，在 vendored 樹只加 3 行攔截 `logits_processor._compute_lm_head`；權重在**載入時於記憶體**量化打包，模型資產完全不動。它是**換路線**，不是改 kernel —— 直接重用生產已驗證的 `gptq_gemm` kernel。

**為何有效**：lm_head 是 248320×5120 的 bf16 矩陣（2.543 GB），每步被讀 **4 次**（verify 1 次 M=4 ＋ draft 3 次 M=1），合共約 13.9 ms/步、佔步時約 20%。M=1 時完全是記憶體綁死，位元寬直接決定成本。

**微基準**（真實 lm_head，gfx1100）：

| | bf16 | int8 | int4 (gs=128) |
|---|---|---|---|
| M=1 | 4.322 ms | 1.803 ms | **0.982 ms（4.40×）** |
| M=4 | 4.437 ms | 6.373 ms | **1.919 ms（2.31×）** |

**端到端**（5K 上下文、temp 0、MTP-3）：bf16 **57.12** → int8 60.43 → int4 **68.38 t/s（+19.7%）**。輸出 MD5 三者**完全相同**、accept length 4.0 不變、U+FFFD 為 0。

### 安裝

    export PYTHONPATH=/path/to/shim:$PYTHONPATH
    export SGL_RDNA_LMHEAD_INT4=1
    export SGL_RDNA_LMHEAD_INT4_GS=128      # 必須 >= 32
    export SGL_RDNA_LMHEAD_INT4_MAXM=8

並依 patch 0004 在 `logits_processor.py` 的 `_compute_lm_head` 開頭加攔截。

### 開關（預設全關 = 零副作用）

| 名稱 | 預設 | 作用 |
|---|---|---|
| `SGL_RDNA_LMHEAD_INT4` | 0 | int4 路徑總開關 |
| `SGL_RDNA_LMHEAD_INT4_GS` | 128 | int4 量化 group size |
| `SGL_RDNA_LMHEAD_INT4_MAXM` | 8 | int4 適用 M 上限 |
| `SGL_RDNA_LMHEAD_INT8` | 0 | int8 路徑總開關 |
| `SGL_RDNA_LMHEAD_INT8_MAXM` | 2 | int8 適用 M 上限 |

### 兩個必須知道的限制

1. **group 16 會靜默產生錯誤結果**（實測相對誤差 92%），故模組強制 group ≥ 32。**32 至 128 之間只實測過 128。**
2. **`M <= 8` 走 INT4、`M > 8` 走 BF16** ⇒ 同一 prompt 的 logits 會因並發批次的組成而異，屬可重現性風險。

---

## `sitecustomize.py` —— 改寫自 AMD-AIM

改寫自 [AMD-AIM/sglang-radeon](https://github.com/AMD-AIM/sglang-radeon)（Apache-2.0）的 `src/sglang_radeon_rdna3/compat.py::_patch_fused_add_rms_norm`：把只接受 4 參數的 `fused_add_rms_norm` 包成同時接受 6 參數形式。**我們只加了包裝與錯誤訊息，未改其語義。** 原專案版權歸 AMD-AIM，於此依 Apache-2.0 轉載並標明出處。
