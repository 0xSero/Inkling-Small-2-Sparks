# =============================================================================
# Inkling-Small-NVFP4 runtime for DGX Spark (GB10 / SM121) — fully pinned.
#
# Layer 0 (Dockerfile.base, unchanged from the original build):
#   pinned base image DIGEST + upstream multidepth-MTP patch (pinned commit +
#   sha256) + pinned pip versions + the image-root/vllm overlay.
#
# Layers below are the two fixes from the 2026-08-17/18 sessions that take this
# from 35.4 -> 60.07 tok/s peak c1. They are applied here as explicit build steps
# so the image is reproducible from source rather than from a mutated container.
# =============================================================================

ARG BASE_IMAGE=vllm/vllm-openai@sha256:cec2df507519b5a2c4d9870524f86a2a100fae79cb689d1ce11f3d4da2f4ffad
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Inkling Small SM121 paged-KV runtime (v38 pinned)" \
      org.opencontainers.image.base.name="vllm/vllm-openai:nightly-65b7662d3fcb773afaf751ab29ac6960a0cf011d" \
      ai.local-studio.vllm.commit="65b7662d3fcb773afaf751ab29ac6960a0cf011d" \
      ai.local-studio.vllm.inkling-multidepth-mtp.commit="0841fed141e9cb18e4464c2b4ec74881cdd90654" \
      ai.local-studio.flash-attention.commit="a80c4d7b5af32426002c42028a66a78408c273b2" \
      ai.local-studio.inkling.speculator-fix="v37" \
      ai.local-studio.inkling.sink-nvfp4="v38"

# --- pinned python deps ------------------------------------------------------
RUN /usr/bin/python3 -m pip install --no-cache-dir \
      av==18.0.0 \
      scipy==1.18.0 \
      soundfile==0.14.0 \
      soxr==1.1.0

# --- upstream inkling multidepth-MTP patch (pinned commit + checksum) --------
ADD --checksum=sha256:fefb0dda787761106709110bf056d8ba42abfbf861e70b27e20a1d9f439dd206 \
    https://github.com/vllm-project/vllm/commit/0841fed141e9cb18e4464c2b4ec74881cdd90654.patch \
    /tmp/inkling-multidepth-mtp.patch

RUN sed -n '/^diff --git a\/vllm\//,$p' /tmp/inkling-multidepth-mtp.patch \
      > /tmp/inkling-multidepth-mtp-runtime.patch \
    && cd /usr/local/lib/python3.12/dist-packages \
    && patch --batch -p1 < /tmp/inkling-multidepth-mtp-runtime.patch \
    && rm /tmp/inkling-multidepth-mtp.patch \
       /tmp/inkling-multidepth-mtp-runtime.patch

# --- inkling model overlay (SM121 paged-KV + inkling model code) -------------
# NOTE: this tree is a required build input. See DEPLOY-README.md "image-root".
COPY image-root/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/

# --- v37: MTP speculator routing fix ----------------------------------------
# init_speculator() routed ALL method=="mtp" to MTPSpeculator, which inherits
# _run_model from AutoRegressiveSpeculator and never passes spec_step_idx. So
# InklingMTP.forward defaulted to spec_step_idx=0 and depth module 0 was reused
# for EVERY draft position instead of cycling 0,1,2. Worth ~+46%.
COPY patches/patch_speculator_fix_v37.py /tmp/patches/
RUN /usr/bin/python3 /tmp/patches/patch_speculator_fix_v37.py

# --- v38: sink-expert NVFP4 W4A16 ------------------------------------------
# Initialises self._unit in process_weights_after_loading() BEFORE the BF16
# weights are freed, and warms the NVFP4 path at every cudagraph capture size so
# Triton JIT finishes before capture begins. Without this the NVFP4 sink branch
# passes alphas=None into sink_silu_mul_epilogue and dies during graph capture.
COPY patches/patch_sink_nvfp4_v38.py /tmp/patches/
RUN /usr/bin/python3 /tmp/patches/patch_sink_nvfp4_v38.py

# --- verify the SM121 paged-KV overlay actually took -------------------------
RUN /usr/bin/python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/ops/fa4_rel_attention.py \
      /usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/ops/qkvr_prep.py \
      /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py \
    && ! grep -R "Paged KV not supported on SM 12.0" \
      /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute
