#!/usr/bin/env python3
"""Quark INT4-W4A16 -> GPTQ-format converter v3 (OFFICIAL semantics).

Verified against AMD Quark source (quark/torch/utils/pack.py, Pack_4_bits):
  pack (pack_method="reorder"): nibble i <- true value at index c*8 + [0,2,4,6,1,3,5,7][i]
  unpack: unpacked[:, ORDER] with ORDER=[0,4,1,5,2,6,3,7]; int4 is SIGNED two's complement
  per_group has a transpose (weight.T before pack), scale shape (K/gs, N)

Target: SGLang exllama gptq_gemm kernel semantics  w = (q_kernel - (qzeros_nibble+1)) * scale
  -> q_kernel = signed4 + 8  (== nibble XOR 8), qzeros nibble = 7  =>  w = signed4 * scale  ✓
"""
import gc, json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QUARK_SRC", "./Qwen3.8-27B-Quark-AWQ-INT4-W4A16")
DST = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GPTQ_DST", "./Qwen3.8-27B-INT4-GPTQ-v3")
os.makedirs(DST, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} SRC={SRC} DST={DST}", flush=True)
BATCH = 120

# official Quark reorder constants
ORDER = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=DEV)   # unpack ORDER
QZEROS_VAL = 0x77777777                                       # kernel: nibble 7 -> zero 8

with safe_open(f"{SRC}/model.safetensors", framework="pt") as sf:
    all_keys = list(sf.keys())
print(f"total tensors: {len(all_keys)}", flush=True)

b8 = torch.arange(8, device=DEV)


def repack_k_major(w_packed_n, K, N):
    """Quark [K, N/8] int32 (N-packed, reordered) -> GPTQ [K/8, N] (K-packed, kernel semantics)."""
    wn = (w_packed_n.unsqueeze(-1) >> (4 * b8)) & 0xF      # [K, N/8, 8] nibble i
    wn = wn[..., ORDER].reshape(K, N)                      # undo official reorder -> true N order
    q_signed = torch.where(wn >= 8, wn - 16, wn)           # signed int4 (two's complement)
    q_kernel = q_signed + 8                                # kernel expects unsigned with zero=8
    qw = torch.zeros(K // 8, N, dtype=torch.int32, device=DEV)
    for i in range(8):
        qw |= (q_kernel[i::8, :] & 0xF) << (4 * i)
    return qw


groups = {}
for k in all_keys:
    base = k
    for suf in (".weight_zero_point", ".weight_scale", ".weight"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    groups.setdefault(base, []).append(k)
items = list(groups.items())
print(f"total groups: {len(items)}", flush=True)

weight_map = {}
shard_idx = 0
n_quant = n_plain = 0
pending = {}

with safe_open(f"{SRC}/model.safetensors", framework="pt") as sf:
    for gi, (base, keys) in enumerate(items):
        wkey = f"{base}.weight"
        has_scale = any(k.endswith("weight_scale") for k in keys)
        if has_scale and wkey in keys:
            w = sf.get_tensor(wkey).to(DEV)
            s = sf.get_tensor(f"{base}.weight_scale").to(DEV)
            z = sf.get_tensor(f"{base}.weight_zero_point").to(DEV)
            K, Np = w.shape
            N = s.shape[1]
            NG = s.shape[0]
            gs = K // NG
            qw = repack_k_major(w, K, N)                     # [K/8, N]
            gidx = torch.arange(K, dtype=torch.int32, device=DEV) // gs
            qzeros = torch.full_like(z, QZEROS_VAL)
            for nm, t in (
                (f"{base}.qweight", qw.cpu()),
                (f"{base}.qzeros", qzeros.cpu()),
                (f"{base}.scales", s.cpu()),
                (f"{base}.g_idx", gidx.cpu()),
            ):
                pending[nm] = t
                weight_map[nm] = None
            n_quant += 1
        else:
            for k in keys:
                pending[k] = sf.get_tensor(k).cpu()
                weight_map[k] = None
            n_plain += 1
        gc.collect()

        if (len(pending) >= BATCH * 3) or gi == len(items) - 1:
            shard_idx += 1
            shard = f"model-{shard_idx:05d}.safetensors"
            save_file(pending, f"{DST}/{shard}")
            for nm in pending:
                weight_map[nm] = shard
            pending = {}
            print(f"  shard {shard_idx} written (quant={n_quant} plain={n_plain})", flush=True)
            gc.collect()

print(f"done: {n_quant} quant repacked, {n_plain} plain", flush=True)
json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map},
          open(f"{DST}/model.safetensors.index.json", "w"), indent=1)

cfg = json.load(open(f"{SRC}/config.json"))
g = cfg.get("quantization_config", {}).get("global_quant_config", {})
wcfg = g.get("weight", {})
cfg["quantization_config"] = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": wcfg.get("group_size", 128),
    "desc_act": False,
    "sym": True,
    "damp_percent": 0.01,
    "true_sequential": False,
    "lm_head": False,
}
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=1)
for fn in os.listdir(SRC):
    if fn.endswith((".json", ".jinja", ".txt", ".model")) and fn not in (
        "config.json", "model.safetensors", "model.safetensors.index.json",
    ):
        shutil.copy(f"{SRC}/{fn}", f"{DST}/{fn}")
print("CONVERT_V3_DONE", flush=True)
