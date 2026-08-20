# Going higher than 64 tok/s c1

Current ceiling and the concrete routes past it. Every number here is measured on
2x DGX Spark (GB10/SM121), not estimated.

## Why config tuning is finished

Throughput obeys one relation, and it fits every measurement to ~1%:

```
tok/s = E * 1000 / step_ms        E = mean accepted tokens/step  <=  num_spec + 1
```

With MTP3, `E <= 4.0`. At the measured ~65 ms step that caps c1 at **61.5 tok/s**,
and we measure **64.01** — i.e. already at the wall, with the small excess coming
from bonus-token accounting.

So c1 cannot improve without either **raising E's ceiling** (more draft positions
that actually get accepted) or **cutting step_ms**. No remaining config knob does
either — the whole ladder is exhausted and its dead ends are in `TUNING-LOG.md`.

## Where the 66.91 ms actually goes

CUDA-event instrumented, live:

```
fwd    = 42.60 ms   target forward, verify batch = 4 tokens
draft  = 19.55 ms   3 MTP depths, serial, ~6.5 ms each
logits =  3.75 ms   201024-vocab lm_head
reject =  0.21 ms
book   =  0.02 ms
total  = 66.91 ms
```

Weight traffic is ~5.3 GB/node/step against ~200-230 GB/s achievable, so the box
runs at **~35-40% of its memory roofline**. The gap is kernel efficiency, not
bandwidth.

## Route 1 — a parallel drafter for Inkling (raises the ceiling)

`RadixArk/Inkling-Small-DSpark` exists on HuggingFace: a DSpark-style drafter for
this exact model. DSpark drafts gamma=5 positions in **one** forward, so the
ceiling becomes `E <= 6.0` instead of 4.0.

Measured on DeepSeek-V4-Flash with gamma=5 on this same hardware: **E = 5.94**,
with 98-100% acceptance at all five positions on predictable text. Inkling's
serial MTP structurally cannot do this — depth `i+1` consumes depth `i`'s hidden
state and sampled token, so the dependency is genuinely sequential.

This is the highest-leverage single change: it lifts the ceiling rather than
chipping at the step.

### ATTEMPTED 2026-08-20 — blocked on vLLM, not on the drafter

The drafter was downloaded and wired in. Two blockers, one solved and one fatal
for *this* vLLM build:

**Solved — target lacked the EAGLE3 interface.** DSpark needs aux hidden states
from target layers `[1,6,12,17,23,28,34,39]`, gated behind vLLM's `SupportsEagle3`
protocol, which Inkling did not implement:

    RuntimeError: Model does not support EAGLE3 interface

`deploy/patches/patch_eagle3_iface_v40.py` adds it — `aux_hidden_state_layers` on
`InklingModel`, capture inside the decoder-layer loop, and the protocol accessors
on `_TmlForCausalLMBase`. It builds and gets past this error. Note the caveat in
the patch header: Inkling defers part of each layer's residual/sconv work via
`pending`, so the captured state is an approximation of what the drafter trained
on. That can only lower acceptance, never correctness.

**Fatal — vLLM's DSpark here is DeepSeek-V4-only.** The registry hardwires it:

```python
"DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM")
```

and that class reads `n_routed_experts`, `index_topk`, `hc_eps`, `hc_mult` —
DeepSeek MoE and Lightning-Indexer fields. The RadixArk checkpoint is a **dense
6-layer Qwen3** drafter with no experts and no indexer. Boot fails walking that
list (`hc_mult`, then `hc_eps`, ...). Supplying the fields would not help: it
would construct a DeepSeek MoE draft architecture that cannot match these weights.

The drafter's own tags say `sglang`/`specforge` — it was trained against a live
SGLang target. Running it needs **either** SGLang, **or** a vLLM-native
`DSparkDraftModel` class for the Qwen3-style dense drafter. That is a new model
implementation, not a config change.

Route 1 therefore remains the highest-value change, but its cost is a vLLM model
port, not a download.

## Route 2 — the draft path (cuts step_ms)

`draft` is 19.55 ms of 66.91 ms. **Correcting an earlier analysis in this repo:**
it is *not* mostly unexplained overhead. Each MTP depth performs a full
201024x4096 vocabulary projection — roughly **785 MiB/rank of BF16 per depth**,
about **2.3 GiB/rank/step** across three. The draft path is largely bandwidth-bound.

Two changes, in order:
1. **Quantise the draft LM head only** (NVFP4/W4A16), keeping the target head at
   BF16. Uses the existing `ModelOptNvFp4W4A16LinearMethod` machinery. A draft-head
   approximation cannot change output correctness — only acceptance — so measure
   top-1 agreement and per-position acceptance after.
2. **TP1-replicate the draft transformer blocks**, keeping the LM head sharded.
   Removes the per-depth cross-node collectives for the cost of duplicated weight
   reads. Note: the existing `patch_mtp_nocomm.py` has its shape commentary wrong
   and affects target layers globally — do not deploy it as-is.

## Route 3 — the MoE forward (biggest single term)

`fwd` is 42.60 ms. The likely cause is **expert-gather locality, not quantisation
format**. At verify batch = 4, the 24 routed assignments spread over nearly 24
distinct experts, so most expert GEMMs have `Mexpert = 1` — a grouped **GEMV**
workload, which cannot saturate bandwidth regardless of weight format.

Prove it before optimising: microbenchmark `flashinfer_cutlass_fused_moe()` with
captured real `topk_ids`, comparing 4 tokens routed to 6 fixed experts vs the same
4 tokens routed to 20-24 distinct experts, at M in {4,8,16,32,64}. If fixed-expert
routing reaches 180-220 GB/s while distinct routing stays near 80 GB/s, locality
is conclusively the gap and W4A16 alone will not fix it.

## What will NOT help

- **More MTP depth.** MTP4 measured: acceptance rises (E 3.0->3.4) but step goes
  69 ms -> 85-137 ms, because MTP-N verifies N+1 tokens and the MoE reads the
  *union* of their top-6-of-256 experts. Net loss.
- **Expert parallel.** On this fabric it trades small TP reductions for 41 layers
  of host-staged cross-node all-to-all, with no GPUDirect RDMA. Wrong topology.
- **A Medusa-style tree.** More verified candidates means a wider expert union,
  and linear MTP4 already loses to that effect.

## Aggregate throughput

Currently **123.0 tok/s at c8**. Aggregate responds to slot count, so the untested
lever is `MAX_NUM_SEQS` 8 -> 16 -> 24, which needs KV headroom, which needs
`GPU_MEMORY_UTILIZATION` above 0.88 (the engine is capped at ~106.5 GB while model
+ KV already use ~103.6 GB). Also worth measuring: whether speculation *reduces*
saturated throughput at high concurrency, since at c8 every step pays three draft
blocks and three vocabulary projections that plain decode does not.

## Operational note

These deployments are owned by `vllm-studio-controller.service`, which restarts
Inkling on boot and generates on the endpoint. Two consequences: config edits can
be reverted underneath you, and benchmarks must gate on an idle engine or step
time inflates (68 ms -> 137 ms observed). Verify config in the **running
container**, never in the env file.
