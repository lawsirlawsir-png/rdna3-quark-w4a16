# 生產補丁系列（gfx1100 / RDNA3）

**基底**：[@StevenChenSE/sglang](https://github.com/StevenChenSE/sglang) 分支 `gfx1100-support` 的 `1442c18`（2026-09-07）
**Patch 數**：19（43 個檔案、+4,097 / −14,870 行）

## 套用

    git clone https://github.com/StevenChenSE/sglang.git
    cd sglang
    git checkout 1442c18
    git am /path/to/patches/*.patch

## 驗證（我們做的，可自行複核）

把本系列套用到 `1442c18` 的**乾淨檢出**，結果與我們生產建置來源**逐位元相同**：

    git clone -s <我們的樹> /tmp/verify && cd /tmp/verify
    git checkout 1442c18
    git am /tmp/series/*.patch
    git diff --quiet <我們的生產 HEAD>      # 無輸出 = 相同

**⇒ 本系列即我們的生產源碼樹，不是節錄。**

## 實質補丁（對效能或正確性有作用者）

| Patch | 作用 | 實測 |
|---|---|---|
| 0001 | gfx1100 生產補丁首次入版控（2026-09-08/09 的改動） | — |
| 0002 | 小 M WMMA 門檻由 M≥16 下移至 `M>=9 && N>=2048` | M=9 GEMM **246.7 → 185.9 µs（−24.6%）**，E2E +1.8~3.9% |
| 0003 | 把不規則（NGRAM）樹排除出 mask-less unified verify kernel | 正確性修正 |
| 0004 | GPTQ 分派補 M_COUNT 5/6/7；lm_head 低位寬攔截點 | M=5 **0.20515 → 0.13743 ms（1.49×）**，M=4/M=8 零成本 |
| 0005 | B2：Σa 預算與 z 修正改為每 group 一次（**必須為編譯期模板**） | M=4 −9.3%／M=5 −8.7%／M=8 −14.2%；E2E **68.38 → 72.84 t/s** |
| 0007 | opt-in N-aware k_split（須設 `SGL_WMMA_KSPLIT_AUTO=1`） | M=16 verify GEMM **−7.9%**，六個生產形狀 fp64 逐位元相同 |
| 0019 | kernel witness 閘門改為 opt-in（`SGL_WITNESS`） | 消除每次 KV 寫入的 GPU→CPU 同步 |

其餘為量測變體隔離、建置整潔、註解與 CU 數更正（96 → 70，96 屬 W7900）。

## 注意

- **0008 體積較大（約 632 KB）**，內容是停止追蹤 21 個由 hipify 生成的 `.hip` 檔；這些檔在 build 時會重新生成。
- 標明 `default path unchanged` 的補丁（0006、0007）預設不改變行為，須以環境變數開啟。
- **B2（0005）必須是編譯期模板。** 第一版用執行期 bool，熱迴圈內的分支破壞 unroll 排程，反而慢 27~84%；改 `if constexpr` 後才得到上表的數字。
- 回滾：`git revert <patch>`；`q_gemm_rdna3.cu` 與其 hipify 生成的 `.hip` 必須同退。

## 出處

- 基底 fork：[@StevenChenSE](https://github.com/StevenChenSE) 的 `gfx1100-support`
- 被修改的 GPTQ / WMMA kernel 源碼來自 [vLLM](https://github.com/vllm-project/vllm)（Apache-2.0，檔案頭保留原版權聲明）
- 本系列中由我們撰寫的改動以 Apache-2.0 釋出
