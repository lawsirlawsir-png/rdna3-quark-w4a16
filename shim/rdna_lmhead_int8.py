# -*- coding: utf-8 -*-
"""RDNA3 lm_head 低位寬路徑（int4 / int8）— 2026-09-12

動機（實測）：lm_head 每步被讀 4 次（verify 1 次 M=4 + draft 3 次 M=1），
bf16 合共 13.9 ms/步 = 20% 步時；M=1 時純讀 2.543 GB，屬記憶體綁死。

實測（真實 lm_head 248320x5120，W7800 gfx1100，group 協議）：
    M=1 : bf16 4.322 ms | int8 1.803 | int4(g128) 0.982   -> int4 = 4.40x
    M=4 : bf16 4.437 ms | int8 6.373 | int4(g128) 1.919   -> int4 = 2.31x
  int4 用生產已驗證的 gptq_gemm kernel（無新 kernel）；權重在載入時於記憶體
  量化打包（套用我們自己轉換器的語義 q_kernel = signed4+8、qzeros = 0x77777777），
  模型資產完全不動。

⚠️ group 16 會靜默產生錯誤結果（實測 rel 92%），故強制 group >= 32。

開關（預設全關 = 零副作用）：
  SGL_RDNA_LMHEAD_INT4=1 / _GS=128 / _MAXM=8
  SGL_RDNA_LMHEAD_INT8=1 / _MAXM=2
"""
import logging
import os
import weakref

import torch

_LOG = logging.getLogger("sglang")


def _warn(msg, *a, **kw):
    """這個模組的每一條回退路徑都必須留痕。

    過去所有失敗都是靜默的：呼叫端的 logits_processor 用 except 吞掉例外，
    這裡再把錯誤計數藏在 _STATS["err"] 裡，結果是 int4 悄悄失效、只退回 bf16，
    效能掉了卻沒有任何訊息。故一律經此函式記錄。
    """
    _LOG.warning("RDNA-LMHEAD " + msg, *a, **kw)


def _env_int(name, default):
    """環境變數解析失敗時用預設值並警告，不讓模組 import 直接崩。

    原寫法 int(os.environ.get(...)) 遇到非整數會在 import 期拋錯；呼叫端
    （logits_processor）的 except 會吞掉，變成靜默回退 bf16。
    """
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        _warn("%s=%r 不是整數，改用預設值 %d", name, raw, default)
        return default


_I4 = os.environ.get("SGL_RDNA_LMHEAD_INT4", "0") == "1"
_I4_GS = _env_int("SGL_RDNA_LMHEAD_INT4_GS", 128)
_I4_MAXM = _env_int("SGL_RDNA_LMHEAD_INT4_MAXM", 8)
_I8 = os.environ.get("SGL_RDNA_LMHEAD_INT8", "0") == "1"
_I8_MAXM = _env_int("SGL_RDNA_LMHEAD_INT8_MAXM", 2)
_ENABLED = _I4 or _I8
_BN = 64
_BK = 256
_QZ = 0x77777777

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception as _e:
    _HAS_TRITON = False
    # 2026-09-13 review：此處原本靜默，令 int8 lm_head 路徑無聲消失，
    # 違反本模組「每一條回退路徑都必須留痕」的原則（見 _warn docstring）。
    _warn("import triton 失敗（%s）；int8 lm_head 路徑停用，int4 不受影響", _e)

if _HAS_TRITON:

    @triton.jit
    def _w8a16_gemv(A, W, S, Out, K,
                    stride_am, stride_ak, stride_wn, stride_wk, stride_om, stride_on,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + m * stride_am + offs_k * stride_ak
        w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs).to(tl.float32)
            w = tl.load(w_ptrs).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        sc = tl.load(S + offs_n).to(tl.float32)
        tl.store(Out + m * stride_om + offs_n * stride_on,
                 (acc * sc).to(Out.dtype.element_ty))


_CACHE = {}
_STATS = {"i4": 0, "i8": 0, "miss": 0, "built": 0, "err": 0, "stale": 0, "strong": 0}


def stats():
    return dict(_STATS)


def _pack_int4(W, gs):
    from sgl_kernel import gptq_shuffle
    N, K = W.shape
    NG = K // gs
    qs, zs, ss = [], [], []
    for i in range(0, N, 8192):
        Wc = W[i:i + 8192].float()
        Wg = Wc.reshape(-1, NG, gs)
        amax = Wg.abs().amax(dim=2).clamp_min(1e-8)
        sc = (amax / 7.0).to(torch.bfloat16)
        q = torch.round(Wg / sc.float().unsqueeze(2)).clamp(-8, 7).to(torch.int8)
        qk = (q.reshape(-1, K) + 8).to(torch.int32)
        qw = torch.zeros((K // 8, Wc.shape[0]), dtype=torch.int32, device=W.device)
        for b in range(8):
            qw |= (qk[:, b::8] & 0xF).t().contiguous() << (4 * b)
        qs.append(qw)
        zs.append(torch.full((NG, Wc.shape[0] // 8), _QZ, dtype=torch.int32, device=W.device))
        ss.append(sc.t().contiguous())
    qw = torch.cat(qs, 1).contiguous()
    qz = torch.cat(zs, 1).contiguous()
    sc = torch.cat(ss, 1).contiguous()
    gi = torch.empty((0,), device=W.device, dtype=torch.int32)
    gptq_shuffle(qw, gi, 4)
    return qw, qz, sc, gi


def _build(W):
    ent = {}
    if _I4:
        gs = _I4_GS
        if gs < 32:
            # 實測：group 16 會「靜默地」產生相對誤差 92% 的結果。這條防線必須存在，
            # 而且必須出聲——否則設錯 env 只會看到速度變慢，看不出結果已錯。
            _warn("group size %d < 32 不受支援（實測 g16 相對誤差 92%%）；"
                  "int4 lm_head 停用，回退 bf16", gs)
        elif W.shape[1] % gs or W.shape[0] % 8 or (W.shape[1] // 8) <= 0:
            _warn("lm_head 形狀 %s 與 group size %d 不相容；"
                  "int4 停用，回退 bf16", tuple(W.shape), gs)
        else:
            ent["i4"] = _pack_int4(W, gs)
    if _I8 and _HAS_TRITON:
        N, K = W.shape
        wb = (W.abs().amax(dim=1).clamp_min(1e-8) / 127.0).float()
        Wq = torch.empty((N, K), dtype=torch.int8, device=W.device)
        for i in range(0, N, 16384):
            j = min(i + 16384, N)
            Wq[i:j] = torch.round(W[i:j].float() / wb[i:j, None]).clamp(-127, 127).to(torch.int8)
        ent["i8"] = (Wq, wb)
    return ent or False


def _owner_ref(o):
    """回傳一個零引數可呼叫物件，回傳原物件；無法弱引用時退回強引用。

    退回強引用會令此條目永久持有該物件（本模組原本的行為），但語義仍正確
    —— 絕不會把過期條目當成本次結果。此情形會計入 _STATS["strong"]，
    不留靜默路徑。
    """
    try:
        return weakref.ref(o)
    except TypeError:
        _STATS["strong"] += 1
        return lambda: o


def _get(lm_head):
    # 2026-09-13 review：以 id() 作快取鍵時，若原物件被回收，CPython 可把同一個
    # id 分配給新物件，命中舊條目就會把「別的張量的打包權重」當成本次結果回傳。
    # 因此每筆條目一併記錄主人的弱引用，命中時必須確認主人仍存活且是同一個物件。
    key = id(lm_head)
    hit = _CACHE.get(key)
    if hit is not None:
        owner, ent = hit
        if owner() is lm_head:
            return ent
        del _CACHE[key]           # id 被重用，丟棄過期條目後重建
        _STATS["stale"] += 1
    W = lm_head.weight
    if W.dim() != 2 or W.dtype != torch.bfloat16 or not W.is_cuda:
        _CACHE[key] = (_owner_ref(lm_head), False)
        return False
    try:
        ent = _build(W.detach())
        _CACHE[key] = (_owner_ref(lm_head), ent)
        _STATS["built"] += 1
    except Exception:
        _CACHE[key] = (_owner_ref(lm_head), False)
        _STATS["err"] += 1
        _warn("int4/int8 打包失敗，此 lm_head 永久留在 bf16", exc_info=True)
        return False
    return ent


def try_lmhead(lm_head, hidden_states):
    """命中回傳 logits，否則 None（呼叫方走原路徑）。"""
    if not _ENABLED:
        return None
    try:
        if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
            _STATS["miss"] += 1
            return None
        M, K = hidden_states.shape
        if M == 0:
            _STATS["miss"] += 1
            return None
        ent = _get(lm_head)
        if not ent:
            _STATS["miss"] += 1
            return None
        A = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        bias = getattr(lm_head, "bias", None)

        if "i4" in ent and M <= _I4_MAXM:
            from sgl_kernel import gptq_gemm
            qw, qz, sc, gi = ent["i4"]
            # 原本只驗 K 與 N。sc 由 _pack_int4 建成 [K // gs, N]，若不驗 group 維，
            # 一旦打包佈局與 kernel 假設不符就會靜默算錯（同 2026-09-10 Quark
            # reorder 事故同類）。K // gs 亦會被 kernel 用來推導 group 邊界。
            if (K == qw.shape[0] * 8 and qw.shape[1] == sc.shape[1]
                    and sc.shape[0] == K // _I4_GS):
                Out = gptq_gemm(A, qw, qz, sc, gi, True, 4)
                if bias is not None:
                    Out = Out + bias
                _STATS["i4"] += 1
                _log()
                return Out

        if "i8" in ent and M <= _I8_MAXM:
            Wq, wb = ent["i8"]
            N = Wq.shape[0]
            if K == Wq.shape[1] and K % _BK == 0 and N % _BN == 0:
                Out = torch.empty((M, N), dtype=A.dtype, device=A.device)
                _w8a16_gemv[(N // _BN, M)](
                    A, Wq, wb, Out, K,
                    A.stride(0), A.stride(1), Wq.stride(0), Wq.stride(1),
                    Out.stride(0), Out.stride(1), BLOCK_N=_BN, BLOCK_K=_BK)
                if bias is not None:
                    Out = Out + bias
                _STATS["i8"] += 1
                _log()
                return Out

        _STATS["miss"] += 1
        _log()
        return None
    except Exception:
        _STATS["err"] += 1
        _warn("lm_head 低位寬路徑拋出例外，回退 bf16", exc_info=True)
        return None


def _log():
    n = _STATS["i4"] + _STATS["i8"] + _STATS["miss"]
    if n and n % 200 == 0:
        # logging 是標準庫，import 不會失敗；原本包一層 except: pass 只是掩蓋問題。
        _LOG.info("RDNA-LMHEAD %s", _STATS)
