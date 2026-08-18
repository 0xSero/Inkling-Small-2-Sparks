#!/usr/bin/env python3
"""Patch v37: Fix MTP speculator routing + sink NVFP4 W4A16.

TWO fixes:
1. Route inkling MTP to MultiModuleMTPSpeculator when use_multi_module_mtp() is true.
   This fixes the critical bug where MTPSpeculator never passes spec_step_idx,
   causing depth 0 to be reused for all draft positions (pos1=39%, pos2=27%).
   With MultiModuleMTPSpeculator, depths 0,1,2 are cycled correctly.

2. Quantize sink experts to NVFP4 W4A16 using flashinfer.mm_bf16_fp4.
   Cuts sink bandwidth from 1.875 GiB to 0.9375 GiB per step.
   No dequant overhead — native W4A16 GEMM via cute-dsl backend.

Built on v34 (MTP cap removed).
"""
import py_compile, sys

# ── Patch 1: Fix speculator routing ────────────────────────────────────────
SPEC_INIT_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py"

with open(SPEC_INIT_PATH) as f:
    init_src = f.read()

# Add MultiModuleMTPSpeculator branch BEFORE the generic mtp branch
OLD_MTP_BRANCH = '''    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)'''

NEW_MTP_BRANCH = '''    elif speculative_config.method == "mtp" and speculative_config.use_multi_module_mtp():
        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        return MultiModuleMTPSpeculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)'''

if OLD_MTP_BRANCH in init_src:
    init_src = init_src.replace(OLD_MTP_BRANCH, NEW_MTP_BRANCH, 1)
    with open(SPEC_INIT_PATH, "w") as f:
        f.write(init_src)
    py_compile.compile(SPEC_INIT_PATH, doraise=True)
    print("SPECULATOR ROUTING FIXED — MultiModuleMTPSpeculator now used for multi-depth MTP")
elif "MultiModuleMTPSpeculator" in init_src:
    print("Speculator routing already patched — skipping")
else:
    print("WARNING: Could not find MTP branch in spec_decode/__init__.py")
    print("Will continue with sink NVFP4 patch anyway")

# ── Patch 2: Sink NVFP4 W4A16 ────────────────────────────────────────────
MOE_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/moe.py"

with open(MOE_PATH) as f:
    moe_src = f.read()

if "sink_nvfp4" in moe_src:
    print("SINK NVFP4 already patched — skipping")
else:
    # 1. Add NVFP4 state to __init__ + process_weights_after_loading
    OLD_INIT = """        self._unit: torch.Tensor | None = None

    def load_weight"""

    NEW_INIT = """        self._unit: torch.Tensor | None = None
        self._sink_nvfp4 = False
        self._w13_packed = None
        self._w2_packed = None

    def process_weights_after_loading(self):
        \"\"\"Quantize BF16 sink expert weights to NVFP4 W4A16.

        Uses flashinfer.mm_bf16_fp4 (cute-dsl backend) for native W4A16 GEMM.
        Controlled by INKLING_SINK_NVFP4 env var (default 1 = enabled).
        \"\"\"
        import os as _os
        if _os.environ.get("INKLING_SINK_NVFP4", "1") != "1":
            return

        try:
            import flashinfer
            # Quantize w13: shape (n_experts, 2*intermediate_pp, d_model)
            # Keep bf16 — nvfp4_quantize only accepts fp16/bf16/e4m3, NOT fp32
            w13 = self.w13_weight.data.to(torch.bfloat16)
            w13_2d = w13.view(-1, w13.shape[-1])  # (E*2F, D)
            absmax13 = w13_2d.abs().max().float()
            if absmax13 > 0:
                gsf13 = (6.0 / absmax13).reshape(1).to(torch.float32)
                w13_q, w13_sf = flashinfer.nvfp4_quantize(
                    w13_2d, gsf13,
                    sfLayout=flashinfer.SfLayout.layout_128x4)
                b_p13, sf_p13, _ = flashinfer.prepare_bf16_fp4_weights(
                    w13_q, w13_sf, backend="cute-dsl")
                alpha13 = (absmax13 / 6.0).reshape(1).to(torch.float32)
                self._w13_packed = (b_p13, sf_p13, alpha13)

            # Quantize w2: shape (d_model, n_experts * intermediate_pp)
            w2 = self.w2_weight.data.to(torch.bfloat16)
            absmax2 = w2.abs().max().float()
            if absmax2 > 0:
                gsf2 = (6.0 / absmax2).reshape(1).to(torch.float32)
                w2_q, w2_sf = flashinfer.nvfp4_quantize(
                    w2, gsf2,
                    sfLayout=flashinfer.SfLayout.layout_128x4)
                b_p2, sf_p2, _ = flashinfer.prepare_bf16_fp4_weights(
                    w2_q, w2_sf, backend="cute-dsl")
                alpha2 = (absmax2 / 6.0).reshape(1).to(torch.float32)
                self._w2_packed = (b_p2, sf_p2, alpha2)

            self._sink_nvfp4 = True
            self.w13_weight.data = torch.empty(0)
            self.w2_weight.data = torch.empty(0)
            print(f"[SINK NVFP4] Quantized sink experts to W4A16", flush=True)
        except Exception as e:
            print(f"[SINK NVFP4] FAILED, falling back to BF16: {e}", flush=True)
            self._sink_nvfp4 = False

    def load_weight"""

    if OLD_INIT in moe_src:
        moe_src = moe_src.replace(OLD_INIT, NEW_INIT, 1)
    else:
        print("Could not find __init__ pattern for InklingSinkExperts")
        sys.exit(1)

    # 2. Replace forward to use NVFP4 GEMM when available
    OLD_FORWARD = """    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        \"\"\"``sum_e gammas[:, e] * MLP_e(x)`` (TP-partial along d_mlp).\"\"\"
        from .ops import sink_silu_mul_epilogue

        # One GEMM over the experts' stacked w13 (a view), fused epilogue,
        # then one GEMM whose K-reduction over the K-concatenated w2 performs
        # the expert sum.
        if self._unit is None or self._unit.device != x.device:
            self._unit = torch.ones(
                self.n_experts, dtype=torch.float32, device=x.device
            )
        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        return h @ self.w2_weight.T  # (T, D)"""

    NEW_FORWARD = """    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        \"\"\"``sum_e gammas[:, e] * MLP_e(x)`` (TP-partial along d_mlp).\"\"\"
        from .ops import sink_silu_mul_epilogue

        if self._sink_nvfp4 and self._w13_packed is not None:
            import flashinfer
            raw = flashinfer.mm_bf16_fp4(
                x, self._w13_packed[0], self._w13_packed[1],
                self._w13_packed[2], backend="cute-dsl")
            h = sink_silu_mul_epilogue(
                raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
            )
            return flashinfer.mm_bf16_fp4(
                h, self._w2_packed[0], self._w2_packed[1],
                self._w2_packed[2], backend="cute-dsl")

        if self._unit is None or self._unit.device != x.device:
            self._unit = torch.ones(
                self.n_experts, dtype=torch.float32, device=x.device
            )
        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        return h @ self.w2_weight.T  # (T, D)"""

    if OLD_FORWARD in moe_src:
        moe_src = moe_src.replace(OLD_FORWARD, NEW_FORWARD, 1)
    else:
        print("Could not find forward pattern for InklingSinkExperts")
        sys.exit(1)

    # 3. Hook process_weights_after_loading into finalize_load
    OLD_FINALIZE_END = """        for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ):
                p = getattr(experts, pname, None)
                if p is not None:
                    p.data[lid].zero_()
        return out"""

    NEW_FINALIZE_END = """        for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ):
                p = getattr(experts, pname, None)
                if p is not None:
                    p.data[lid].zero_()
        self.sink_experts.process_weights_after_loading()
        return out"""

    if OLD_FINALIZE_END in moe_src:
        moe_src = moe_src.replace(OLD_FINALIZE_END, NEW_FINALIZE_END, 1)
    else:
        print("WARNING: Could not find finalize_load end — hook NOT added")

    with open(MOE_PATH, "w") as f:
        f.write(moe_src)

    py_compile.compile(MOE_PATH, doraise=True)
    print("SINK NVFP4 W4A16 PATCHED")

print("=== PATCH V37 COMPLETE ===")
print("Fix 1: MultiModuleMTPSpeculator routing (fixes MTP3 depth cycling)")
print("Fix 2: Sink NVFP4 W4A16 GEMM (cuts sink bandwidth 50%)")
