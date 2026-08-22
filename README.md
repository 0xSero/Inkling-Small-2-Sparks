# Inkling-Small-2-Sparks

Tuned, pinned, reproducible serving of **Inkling-Small-NVFP4 across 2× NVIDIA DGX Spark**
(GB10 / SM121) with vLLM, TP2 over RoCEv2, and multi-token-prediction speculative decoding.

```bash
git clone https://github.com/0xSero/Inkling-Small-2-Sparks && cd Inkling-Small-2-Sparks
cp deploy/env.example deploy/inkling.env   # fill in host addresses + model path
./run.sh
```

That brings the stack up on both nodes with the fast config and smoke-tests it. Then:

```bash
./run.sh chat "Explain consistent hashing."
```

## Measured results

2× DGX Spark (GB10, 128 GB unified LPDDR5X each, 273 GB/s per node), TP2 over a
200 GbE ConnectX-7 direct link (RoCEv2, MTU 9000). Idle-gated, contention-free.

| metric | value |
|---|---|
| decode c1 | **46.78 tok/s median · 64.01 tok/s peak** |
| step time | **64.3 – 66.2 ms** (constant across prompts) |
| aggregate @ c48 | **367.4 tok/s** |
| max context | **262,144** (verified serving 234,221 prompt tokens) |
| coherence | 0 repeated 8-grams, natural EOS on all probes |

Concurrency sweep, 384 tok/stream (each row measured at a slot count ≥ its
concurrency, with that batch size CUDA-graph-captured):

| conc | per-stream median | aggregate |
|---|---|---|
| 1 | 34.6 | 34.6 |
| 2 | 31.2 | 62.4 |
| 4 | 19.7 | 78.9 |
| 8 | 17.9 | 125.8 |
| 12 | 13.1 | 156.9 |
| 16 | 14.9 | 191.1 |
| 24 | 11.6 | 247.0 |
| 32 | 9.5 | 272.6 |
| 40 | 8.4 | 305.1 |
| 48 | 8.5 | **367.4** |
| 64 | 7.4 | 358.0 — saturated |

Two things made the aggregate curve real:

1. **Slot count must cover the concurrency.** With 4 slots, a c8 run queues half
   its requests and aggregate falls *below* c4 (measured: 82.2). Every row above
   was taken with `MAX_NUM_SEQS` ≥ the concurrency.
2. **The batch size must be CUDA-graph-captured.** c12 measured 115.0 tok/s when
   batch 12 fell outside the capture list, and 156.9 once `12` was added
   (+36% from capture alone). The shipped config captures
   `[1,2,3,4,8,16,32,40,48]`.

Saturation is at **c48 ≈ 360–367 tok/s**; c64 gains nothing and per-stream
drops to 5.6 tok/s minimum. c1 decode is unaffected by the 48-slot config
(steps stay 65–68 ms).

## Second profile: 1M context (SGLang + DSpark + FP4 KV)

The vLLM profile above is the **throughput** champion. There is a second,
independently baked profile for **latency + long context**, built on the
public `lmsysorg/sglang` GB10 dev image plus the patch set in
[0112358DEUS/inklingdeus](https://github.com/0112358DEUS/inklingdeus)
(`KVQUANT=1 ./scripts/bake-image.sh`), serving the same NVFP4 weights with the
RadixArk-trained DSpark draft head
(`RadixArk/Inkling-Small-DSpark-Preview`):

| metric | vLLM profile (above) | SGLang 1M profile |
|---|---|---|
| max context | 262,144 | **1,048,576** (KV pool 1,319,267 tok, `fp4_mx_block16`) |
| decode c1 peak | 64.0 | **85.9** |
| decode c1 median | 46.8 | 44.4 – 47.5 |
| aggregate | **367.4** @ c48 | ~109 @ c16 |
| coherence | clean | clean (needle retrieved exactly from a 307,581-token prompt (prefill ~657 tok/s at that depth)) |

Launch: `BLOCK=10 MAXREQ=48 ./scripts/nvfp4-kv-boot.sh <rank>` (worker rank 1
first). DSpark block-size sweep on this head (trained for block ≤ 15):
peak climbs 45.8 → 80.7 → **85.9** across block 5 → 7 → 10, then median
collapses at 15 (35.6) from overdrafting — and large blocks destroy batch
throughput (aggregate pinned at ~55–61 tok/s at any concurrency), which is why
the throughput profile stays on vLLM/MTP3.

## How it got fast

Starting point was 35.4 tok/s. Three changes, each measured:

1. **MTP speculator routing fix** (+46%) — `init_speculator()` routed every
   `method=="mtp"` config to `MTPSpeculator`, which inherits `_run_model` from
   `AutoRegressiveSpeculator` and never passes `spec_step_idx`. `InklingMTP.forward`
   therefore defaulted to `spec_step_idx=0`, reusing depth module 0 for *all* draft
   positions instead of cycling 0,1,2. → `deploy/patches/patch_speculator_fix_v37.py`
2. **NVFP4 sink experts** (+9%) — halves sink-expert bandwidth. Needs `self._unit`
   initialised before the BF16 weights are freed, plus a warmup at every cudagraph
   capture size so Triton JIT finishes before capture.
   → `deploy/patches/patch_sink_nvfp4_v38.py`
3. **FlashInfer autotune** (+7% step time) — was hard-disabled. The older
   "backend choice is irrelevant at decode" result was measured at M≤2; MTP3 makes
   the verify GEMM M=4, so it no longer applies.

## The governing equation

With correct measurement, **step time is constant regardless of prompt**, so

```
tok/s = E × 1000 / step_ms          (E = mean MTP acceptance, ≤ num_spec + 1)
```

fits every measured row to ~1%. Two consequences worth knowing before tuning:

- **MTP3 ceiling is `4.0 / step`.** At 66.2 ms that is 60.4 tok/s, and we measure
  64.01 at E=3.98. On predictable prompts there is provably nothing left at MTP3.
- **More draft depth is not free.** MTP-N verifies N+1 tokens and the MoE reads the
  *union* of their top-6-of-256 experts, so weight traffic grows with depth. MTP4
  measured: acceptance up (E 3.0→3.4), step time 69 → 85–137 ms. Net loss.

## Measure it yourself

```bash
./run.sh bench
```

Two things the harness does that matter. It **settles** the Prometheus spec-decode
counters (they flush on an interval, so naive deltas smear one request's acceptance
into the next — this produced impossible rows like "E=3.47 but 33 tok/s"), and it
**gates on an idle engine** (other traffic on the endpoint inflates step time
68 → 137 ms). Numbers taken without both are wrong.

Benchmarks read `INKLING_URL`, so they can be driven from the second node.

## Judging quality

Use `/v1/chat/completions`. Raw `/v1/completions` makes a chat model *continue* the
prompt — which reads as echo/repetition and looks like degeneration but isn't — and
silently caps at the OpenAI default of 16 tokens. On the chat path, temp 0, allowed
to stop naturally, all probes finished at their own EOS with zero repeated 8-grams.

## Everything is pinned

Base image by **digest**, not tag; vLLM commit `65b7662d`; flash-attention
`a80c4d7b`; the upstream multidepth-MTP patch by commit `0841fed1` *and* sha256;
pip versions; arch `12.1a`/`sm_121a`. See `deploy/Dockerfile` and
`docs/DEPLOYMENT.md`.

## Guardrails

These were learned by breaking hardware, so they are not stylistic advice:

- **Change one variable per restart.** Stacking `MAX_MODEL_LEN` + `KV_CACHE_MEMORY_BYTES`
  + `MAX_NUM_SEQS` overcommitted host RAM during engine init; the OOM killer took
  userspace including sshd and the node needed a physical power-cycle.
- **`KV_CACHE_MEMORY_BYTES` floor ≈ 18.5 GB.** At 16 GB the engine refuses to boot
  (262144 ctx needs 17.3 GiB; 16 GB yields 14.9 GiB).
- **Clean `/dev/shm/psm_*` on both nodes after any crash**, or NCCL rendezvous fails
  with a TCPStore broken pipe on rank 1. `run.sh` does this for you.
- **Give sshd `OOMScoreAdjust=-1000`.** The box runs ~113/121 GB while serving and
  sshd gets killed during model load, costing shell access on a live node.
- **`flashinfer_cutlass` is the only working MoE backend on SM121.** `flashinfer_trtllm`
  refuses to boot (unquantized sink path); `flashinfer_b12x` and `flashinfer_cutedsl`
  JIT-storm and wedge the node; `marlin` boots but is only +1.5%.

## Known-open

- **c8 unmeasured** — needs `MAX_NUM_SEQS=8`, changed alone.
- **Context > 262144** is KV-bound, not model-bound (`model_max_length` is 1048576).
  262144 costs 17.3 GiB, so ~26 GiB ≈ ~390k. Walk it up in steps.
- **Past 60 tok/s c1** needs either the draft path (19.55 ms of 66.91 ms, ~85% of it
  explained by neither weight traffic nor kernel launches — cross-node collectives
  inside the captured graph are the prime suspect) or a W4A16 MoE decode path.

## Hillclimbing

`bench/hillclimb.py` walks a ladder of single-variable config changes, measuring
decode, prefill and concurrency at each step and keeping a change only if it beats
the current best. Coherence is a **hard gate** — an incoherent config scores -1 and
is reverted regardless of how fast it is, as is any run that detected contention.

```bash
INKLING_HEAD=<head> INKLING_WORKER=<worker> python3 bench/hillclimb.py measure  # no changes
INKLING_HEAD=<head> INKLING_WORKER=<worker> python3 bench/hillclimb.py step     # one candidate
```

It changes **exactly one variable per restart** and refuses candidates that would
push KV below the ~18.5 GB boot floor, or ask for more context than the KV budget
covers (`ctx x 17.3GiB/262144`, halved for fp8 KV). Memory-raising candidates are
deferred unless the host has >=12 GB free. These rails are not decoration: stacking
three memory changes in one restart is what OOM-killed a node's userspace and cost a
physical power-cycle.

State in `hillclimb_state.json`, full history in `hillclimb_history.jsonl`.

## Going higher

64.01 tok/s c1 is the MTP3 wall, not a tuning shortfall: `tok/s = E * 1000/step`
with `E <= 4.0` caps c1 at 61.5 at the measured 65 ms step. Routes past it —
a gamma=5 parallel drafter, the draft LM-head quantisation, and the MoE
expert-gather fix — are in [docs/GOING-HIGHER.md](docs/GOING-HIGHER.md).

## Layout

```
run.sh                 one-command bring-up / chat / bench / status / stop
deploy/Dockerfile      pinned, reproducible build
deploy/docker-compose.yml
deploy/env.example     every tuned value, annotated with why
deploy/patches/        the two source patches
bench/                 decode, prefill, concurrency, context harnesses
bench/hillclimb.py     single-variable hillclimb driver, coherence-gated
docs/DEPLOYMENT.md     full deployment + recovery notes
docs/TUNING-LOG.md     what was tried, what worked, what failed and why
```

## License

Apache-2.0 for the code in this repository. The base runtime is vLLM (Apache-2.0);
model weights are **not** distributed here and are mounted at runtime.
