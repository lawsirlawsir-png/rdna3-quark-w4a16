# RDNA3 compat shim (loaded by EVERY python process, incl. sglang spawn children).
# Source: AMD-AIM/sglang-radeon src/sglang_radeon_rdna3/compat.py _patch_fused_add_rms_norm
try:
    import vllm._custom_ops as _ops
    _real = getattr(_ops, "fused_add_rms_norm", None)
    if _real is not None and not getattr(_real, "_rdna3_wrapped", False):
        def _fused_add_rms_norm(*args):
            if len(args) == 4:
                return _real(*args)
            if len(args) == 6:
                out, x, residual_out, residual, weight, eps = args
                out.copy_(x)
                residual_out.copy_(residual)
                _real(out, residual_out, weight, eps)
                return None
            raise TypeError("fused_add_rms_norm() got %d positional arguments" % len(args))
        _fused_add_rms_norm._rdna3_wrapped = True
        _ops.fused_add_rms_norm = _fused_add_rms_norm
except Exception:
    pass
