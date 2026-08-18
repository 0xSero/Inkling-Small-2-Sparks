#!/usr/bin/env python3
"""Patch v38: Fix sink NVFP4 W4A16 Triton crash (v37 bug).

ROOT CAUSE (found in v37 crash log):
  In the NVFP4 forward branch, `self._unit` is passed as `alphas` to
  `sink_silu_mul_epilogue` BEFORE it is initialized. The BF16 branch
  initializes `self._unit` lazily on first call, but the NVFP4 branch
  uses it without that lazy init, so it's `None`. The Triton kernel then
  receives `None` for `alpha_ptr`, causing:
    AttributeError("'NoneType' object has no attribute 'type'")
  during CUDA graph capture.

FIX:
  1. Initialize `self._unit` in `process_weights_after_loading()` so it
     exists before any forward call.
  2. Add eager warmup of the full NVFP4 forward path
     (mm_bf16_fp4 -> sink_silu_mul_epilogue -> mm_bf16_fp4) at the end
     of `process_weights_after_loading()`, with a small dummy input, to
     trigger Triton JIT compilation BEFORE CUDA graph capture begins.
     This is the standard pattern for Triton kernels under CUDA graphs.

Built on v37 (which has the speculator fix already applied).
"""
import py_compile

MOE_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/moe.py"

with open(MOE_PATH) as f:
    moe_src = f.read()

if "sink_nvfp4_v38" in moe_src:
    print("SINK NVFP4 v38 already patched — skipping")
else:
    # ── Fix 1: Initialize _unit in process_weights_after_loading ─────────────
    # Find the line `self._sink_nvfp4 = True` inside process_weights_after_loading
    # and add _unit init + warmup right after it.
    OLD_NVFP4_FLAG = """            self._sink_nvfp4 = True
            self.w13_weight.data = torch.empty(0)
            self.w2_weight.data = torch.empty(0)
            print(f"[SINK NVFP4] Quantized sink experts to W4A16", flush=True)"""

    NEW_NVFP4_FLAG = """            self._sink_nvfp4 = True
            # Initialize _unit now so the NVFP4 forward branch has it ready.
            # (The BF16 branch inits lazily, but NVFP4 branch uses it first.)
            self._unit = torch.ones(
                self.n_experts, dtype=torch.float32,
                device=w13.device,
            )
            # Save shape info before freeing the BF16 weights.
            self._d_model = w13.shape[-1]
            self._intermediate_pp = w2.shape[1] // self.n_experts
            self._device = w13.device
            self.w13_weight.data = torch.empty(0)
            self.w2_weight.data = torch.empty(0)
            print(f"[SINK NVFP4] Quantized sink experts to W4A16", flush=True)
            # sink_nvfp4_v38: warmup kernels BEFORE cudagraph capture
            self._warmup_nvfp4()"""

    if OLD_NVFP4_FLAG in moe_src:
        moe_src = moe_src.replace(OLD_NVFP4_FLAG, NEW_NVFP4_FLAG, 1)
    else:
        print("ERROR: Could not find NVFP4 flag block — aborting")
        import sys; sys.exit(1)

    # ── Fix 2: Add _warmup_nvfp4 method ──────────────────────────────────────
    # Insert it right before `def load_weight`.
    OLD_LOAD_WEIGHT = """    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:"""

    NEW_LOAD_WEIGHT = """    def _warmup_nvfp4(self):
        \"\"\"Warm up NVFP4 GEMM + sink_silu_mul_epilogue kernels.

        Triggers Triton JIT compilation and CuTeDSL kernel building BEFORE
        CUDA graph capture begins. Without this warmup, the first forward
        inside a graph-capture region triggers compilation, which fails
        because Triton cannot compile while a graph is being captured.
        \"\"\"
        import flashinfer
        from .ops import sink_silu_mul_epilogue

        d_model = self._d_model
        intermediate_pp = self._intermediate_pp
        s2f = self.n_experts * 2 * intermediate_pp
        device = self._device

        # Warmup at each cudagraph capture size used by the sink path.
        # Sink experts are called with T = num_tokens (same as main model).
        for t in (1, 2, 3, 4, 6, 8, 16):
            x = torch.zeros(t, d_model, device=device,
                            dtype=torch.bfloat16)
            gammas = torch.ones(t, self.n_experts, dtype=torch.float32,
                               device=device)
            raw = flashinfer.mm_bf16_fp4(
                x, self._w13_packed[0], self._w13_packed[1],
                self._w13_packed[2], backend="cute-dsl")
            h = sink_silu_mul_epilogue(
                raw, self._unit, gammas, self._unit,
                self.n_experts, torch.bfloat16)
            out = flashinfer.mm_bf16_fp4(
                h, self._w2_packed[0], self._w2_packed[1],
                self._w2_packed[2], backend="cute-dsl")
            del x, gammas, raw, h, out
        torch.cuda.synchronize()
        print(f"[SINK NVFP4] Warmup done for t in (1,2,3,4,6,8,16)", flush=True)

    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:"""

    if OLD_LOAD_WEIGHT in moe_src:
        moe_src = moe_src.replace(OLD_LOAD_WEIGHT, NEW_LOAD_WEIGHT, 1)
    else:
        print("ERROR: Could not find load_weight to insert warmup method")
        import sys; sys.exit(1)

    # ── Fix 3: In forward NVFP4 branch, ensure _unit exists (defensive) ────
    OLD_FWD_NVFP4 = """        if self._sink_nvfp4 and self._w13_packed is not None:
            import flashinfer
            raw = flashinfer.mm_bf16_fp4("""

    NEW_FWD_NVFP4 = """        if self._sink_nvfp4 and self._w13_packed is not None:
            import flashinfer
            if self._unit is None or self._unit.device != x.device:
                self._unit = torch.ones(
                    self.n_experts, dtype=torch.float32, device=x.device)
            raw = flashinfer.mm_bf16_fp4("""

    if OLD_FWD_NVFP4 in moe_src:
        moe_src = moe_src.replace(OLD_FWD_NVFP4, NEW_FWD_NVFP4, 1)
    else:
        print("WARNING: Could not find forward NVFP4 branch to patch _unit guard")

    with open(MOE_PATH, "w") as f:
        f.write(moe_src)

    py_compile.compile(MOE_PATH, doraise=True)
    print("SINK NVFP4 v38 PATCHED — _unit init + warmup added")

print("=== PATCH V38 COMPLETE ===")
print("Fix: Initialize _unit in process_weights_after_loading() + warmup kernels")
print("This resolves the Triton 'NoneType' crash during CUDA graph capture")
