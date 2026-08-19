# Session 2026-08-18: measurement fix + autotune win + the real wall

## TL;DR

Prior session's numbers were **measured wrong**. Re-measured with a correct instrument:

| config | c1 median | c1 peak | step time |
|---|---|---|---|
| v38 baseline (autotune=False) | 39.6 | 44.3 | ~71ms |
| **v38 + flashinfer autotune (current)** | **42.5** | **57.7** | **~69ms** |
| MTP4 + autotune (tested, rejected) | ~32 | 58.1 | 85–137ms |

## The measurement bug

`SESSION-20260817-50TOKS.md` scraped Prometheus spec-decode counters immediately
around each request. vLLM flushes those counters on an interval, so each request's
delta absorbed the previous request's tail. That produced impossible rows like
"E=3.47 but only 33 tok/s".

Fixed in `bench_decode_v39.py`: 2s settle on both sides of every request. After the
fix the data is self-consistent — **step time is essentially constant (~69ms)
regardless of prompt**, and

    tok/s = E x 1000 / step_ms

fits every measured row to within ~1%. Throughput is governed *entirely* by MTP
acceptance E, not by anything prompt-dependent in the forward pass.

## Second measurement bug: foreign traffic

The endpoint is shared — a `vllm-studio` controller and/or a human uses it while
benchmarks run. `num_requests_running=1` with no bench active was observed directly.
Contended requests share the GPU and inflate step time (68ms -> 105-137ms), which is
what made the MTP4 and restore runs look erratic.

`bench_decode_v39.py` now gates on idle (`wait_idle()`) before each measurement and
flags any request that became contended mid-flight with `!! CONTENDED`. **Do not
trust any number in this repo that was taken without that gate.**

## What worked: flashinfer autotune

`docker-compose.yml:162` hard-disabled it (`--kernel-config.enable_flashinfer_autotune=False`).
The DSV4-flash reference recipe enables it. The old "backend choice is irrelevant at
decode" finding was measured at M<=2; MTP3 makes the verify GEMM M=4, so that finding
no longer applies.

Enabling it cut step time ~71ms -> ~66-69ms uniformly across every prompt (a
kernel-level win, not acceptance noise). Median 39.6 -> 42.5, peak 44.3 -> 57.7.
Costs ~3-4 min extra boot time for the tuning sweep.

## What failed: MTP4

Acceptance improved a lot (E 3.0-3.4 vs 2.3-3.0) but step time went 69ms -> 85-137ms.
Net loss. Cause: step cost scales **superlinearly with verify-batch size** — MTP-N
verifies N+1 tokens, and the MoE reads the *union* of the experts those tokens route
to (256 experts, top-6, so up to 6(N+1) distinct experts per layer). More depth buys
acceptance but pays for it in weight traffic. This is the central constraint.

## The wall (why 65 tok/s c1 is not reachable by config)

MTP3 can accept at most 4 tokens/step, so its ceiling is `4.0 / step`:

- at the measured ~69ms step -> **58.0 tok/s absolute ceiling**
- we already measure **57.7 peak** — i.e. on predictable prompts we are *at* the
  MTP3 ceiling, with ~98% acceptance at every position. There is nothing left there.
- 65 tok/s would need step <= 61.5ms *and* perfect acceptance simultaneously.
- On diverse prompts (E ~ 2.2-3.0), 65 tok/s needs a **37-46ms step** — a 35-45% cut.

MTP4/5 raise the E ceiling but raise step time faster (measured above), so they move
away from the target rather than toward it.

## Where the 69ms actually goes

Roughly: fwd ~45ms, draft ~18ms (3 serial MTP modules), logits ~3.5ms.

Weight traffic per node per step, computed from `config.json` (42 layers = 41 MoE + 1
dense, 256 routed experts, top-6, expert intermediate 2048, 2 sink experts at
intermediate 16384, vocab 201024, NVFP4 = 0.5 B/param):

- routed experts  ~3.9 GB  (dominant; scales with verify-batch expert union)
- lm_head         ~0.82 GB
- attention       ~0.44 GB
- sinks           ~0.10 GB (after the v38 NVFP4 fix halved this)

~5.3 GB / 69ms = **~80 GB/s effective, against ~200-230 GB/s achievable on GB10**.
So we run at roughly 35-40% of memory-bandwidth roofline. The roofline does *not*
forbid 65 tok/s — kernel efficiency does.

## The only real paths to 65

1. **W4A16 MoE decode path** (port the DSV4/b12x concept to FusedMoE). This is what
   makes DSV4-flash hit 52-54 on the same two Sparks. Biggest lever by far, but the
   b12x/CuTeDSL route has wedged these nodes with JIT storms twice before.
2. **Cut draft cost** (~18ms of 69ms, 3 serial modules). Replicating draft heads TP1
   removes cross-node collectives per draft depth. ~4 NCCL ops x 3 depths today.
3. **Acceptance calibration** — p1/p2 collapse to 37%/21% on explanatory prompts.
   Raising those is worth more than any remaining config change.

Note 1 and 2 are code work, not configuration. Configuration is exhausted.

## Hardware / fleet (verified this session)

Exactly **2 reachable DGX Sparks**, both running the same stack in TP2:
node A (head, rank 0) + node B (worker, rank 1), `local/inkling-vllm:v38-20260817`.
(Other nodes in the lab were not part of this deployment.)

Interconnect is correct and not a bottleneck: RoCEv2 over `enp1s0f1np1` @200G, MTU
9000, `NCCL_IB_HCA=rocep1s0f1` matching the PORT_ACTIVE device. The second 200G port
(`enP2p1s0f1np1`, MTU 1500) is idle — carries ~8 MB vs 362 GB on the active one.

## Incident

Mid-session the stack died on both nodes (GPU OOM, `NV_ERR_NO_MEMORY`) under
benchmark + foreign traffic contention. Recovery needs `/dev/shm/psm_*` +
`sem.mp-*` cleaning or NCCL rendezvous fails with TCPStore broken-pipe on rank 1.

**KV floor:** `KV_CACHE_MEMORY_BYTES` must be >= ~18.5 GB. At 16 GB the engine
refuses to boot: max_model_len 262144 needs 17.3 GiB and 16 GB yields 14.9 GiB
usable. 20 GB is the known-good value — do not trim it without also cutting
`MAX_MODEL_LEN`.

## Current live config

```
Image: local/inkling-vllm:v38-20260817
MTP_NUM_TOKENS=3
MOE_BACKEND=flashinfer_cutlass
--kernel-config.enable_flashinfer_autotune=True   <-- this session
INKLING_SINK_NVFP4=1
INKLING_SINK_FP8=1
INKLING_DECOMP=1
KV_CACHE_MEMORY_BYTES=20000000000
GPU_MEMORY_UTILIZATION=0.88
```

## Files

- `bench_decode_v39.py` — settled-metrics + idle-gated decode benchmark (use this one)
- `exp2_v39.sh` — config experiment runner (exp_v39.sh v1 buffered output, don't use)
- `results_v39.log` — raw results

## Addendum: backend + draft-path findings

- **flashinfer_trtllm: REJECTED.** Boots fail with
  `ValueError: Unquantized MoE backend FlashInfer TRTLLM does not support the
  deployment configuration since kernel does not support current device cuda.`
  It's the *unquantized sink-expert* path that rejects it, same failure class as
  b12x/cutedsl. With cutlass working, marlin marginal, b12x wedging, cutedsl
  storming and trtllm unsupported, **MoE backend config is exhausted on SM121.**

- **Live decomp on the delivered config:**
  `fwd=42.60 logits=3.75 reject=0.21 book=0.02 draft=19.55 total=66.91`

- **Draft is not bandwidth-bound and not launch-bound.** It is CUDA-graph captured
  (boot log: "Capturing multi-module MTP CUDA graphs (FULL)", 3 graphs = 3 depths),
  so launch overhead is already collapsed. Each depth is a BF16 *unquantized*
  (`quant_config=None`) dense-MLP block (`force_dense_mlp=True`, intermediate 16384)
  ≈243 MB/node/depth → ~0.9ms of weight traffic at 273 GB/s, against **6.52ms
  measured**. So ~85% of draft time is neither weights nor launches.

  Suspect: cross-node collectives inside the captured graph. Each depth runs
  `RowParallelLinear` in both attention o_proj and MLP down_proj (1 all-reduce each),
  and `short_conv.py:95` notes replicated ATTN/MLP sconv streams take
  `x is (T, full_H) from all_reduce` — more collectives per depth.
  Python-level probes cannot see these (graph replay bypasses Python), and the
  existing comm-v27 probe in `communication_op.py` never fires, so the model does not
  route through that wrapper.

- **Highest-value remaining work, in order:**
  1. **TP1-replicate the draft blocks.** Removes every cross-node collective from the
     draft path; costs +0.9ms/depth of duplicated weight reads. If comm is the 5.6ms,
     draft goes 19.6ms -> ~5ms and step 67 -> ~52ms, i.e. **~65 tok/s on
     predictable prompts and ~48-50 median.** This is the single change that could
     reach the target.
  2. W4A16 MoE decode path for `fwd` (42.6ms, the biggest term) — the DSV4-flash
     approach. Largest ceiling, highest risk (this is the b12x/JIT-storm route).
  3. Acceptance calibration — p1/p2 collapse to 40%/30% on explanatory prompts.

  Both 1 and 2 are code changes, not configuration.

## Warning: the endpoint is in active use

The final verification run had **3 of 6 prompts flagged `!! CONTENDED`** — a real
client was generating on the endpoint at the time. Uncontended rows in that run
(step 66.5-67.9ms) match the clean earlier run; contended rows (78-106ms) do not.
Any further restart-based experiments interrupt a live user.

## Post-reboot results (clean machine, no vllm-studio contention)

After the head was power-cycled and the fatal config reverted before auto-start,
the same config measures **better** — the `vllm-studio` controller was no longer
running, so nothing else was sharing the GPU:

| | c1 median | c1 peak | step |
|---|---|---|---|
| before reboot (contended box) | 42.5 | 57.7 | 66-69ms |
| **after reboot (clean box)** | **43.0** | **60.07** | **64.3-66.2ms** |

peak 60.07 at E=3.98 / step 66.2ms is exactly the MTP3 ceiling (4.0/0.0662 = 60.4).
On predictable prompts there is provably nothing left at MTP3.

## Concurrency (clean box, 384 tok/stream)

| conc | per-stream med | aggregate |
|---|---|---|
| c1 | 37.3 | 37.3 |
| c2 | 35.3 | 61.5 |
| c4 | 27.3 | **94.1** |
| c8 | 20.5 | 82.2  <- NOT a real c8: MAX_NUM_SEQS=4 means 4 run + 4 queue |

**Aggregate throughput peaks at c4 = 94.1 tok/s.** c8 still needs
`MAX_NUM_SEQS>=8` to be a valid measurement — change that variable ALONE.

## Coherence: GOOD — and the prior "degeneracy" finding was wrong

`SESSION-20260817-50TOKS.md` claimed "the base model produces degenerate repetitive
output for some creative prompts at temperature=0 (pre-existing model behavior)".
**That was endpoint misuse, not a model defect.** Those tests used raw
`/v1/completions`, where a chat/reasoning model simply *continues* the prompt text —
which looks like echo/repetition. (It also silently capped at the OpenAI default
`max_tokens=16`.)

On the real path — `/v1/chat/completions`, temperature 0, allowed to stop naturally:

| prompt | tokens | finish | repeated 8-grams | tok/s |
|---|---|---|---|---|
| code | 926 | stop | 0 | 44.9 |
| explain | 1334 | stop | 0 | 36.8 |
| story | 2058 | stop | 0 | 32.5 |
| list | 1943 | stop | 0 | 36.4 |
| reason | 424 | stop | 0 | 44.8 |

Every one terminated at its own EOS with **zero** repeated 8-grams, and the content
is correct and well-structured (working two-pointer merge, accurate B+ tree
explanation, coherent narrative). Real-workload chat throughput is 32.5-44.9 tok/s,
consistent with the c1 decode numbers.

**Use `/v1/chat/completions` to judge quality on this model. Raw completions will
mislead you.**

## sshd keeps dying on the head under memory pressure

After the reboot inkling started fine and serves on :8000, but **sshd died again**
during model load (ports: 22 closed, 8000 OPEN, box pings at 0.57ms). The box runs
~113/121 GB while serving and the OOM killer takes sshd despite earlyoom's
`--avoid` list.

Workaround used here: drive everything from the worker against the head's :8000
(`INKLING_URL=http://<head>:8000 python3 bench_decode_v39.py ...`). Both benchmark
scripts now read `INKLING_URL` from the environment.

This is worth fixing independently — add an OOM score adjustment for sshd, or leave
more headroom via `GPU_MEMORY_UTILIZATION`.

## Max context — measured on the live deployment

`served max_model_len: 262144`, and it genuinely serves it:

| target | actual prompt tokens | end-to-end wall (64 tok out) |
|---|---|---|
| 16k | 20,007 | 8.3s |
| 64k | 77,351 | 23.9s |
| 128k | 153,805 | 40.8s |
| 200k | **234,221** | 55.5s |
| 250k | HTTP 400 — exceeds 262144 |

**Max usable context today: 262,144 tokens, verified serving 234k.**

Caveat on those numbers: the wall time is dominated by *prefill*, so tok/s computed
from it is end-to-end latency, not decode rate — prefilling 234k tokens takes ~55s.
Decode rate after prefill is unchanged.

Going higher needs KV, not model changes (`model_max_length` is 1048576): 262144 ctx
costs 17.3 GiB KV, so ~26 GiB ≈ ~390k. Walk it up **one variable at a time** —
stacking ctx + KV + seqs in one restart is what OOM-killed the head.

## 2026-08-19 — MAX_NUM_SEQS 4 -> 8 (ACCEPTED, hillclimb candidate 1)

First automated hillclimb result. Score 0.9070 -> 1.0801.

| metric | 4 slots | 8 slots |
|---|---|---|
| decode c1 median | 40.64 | **47.12** |
| decode c1 peak | 60.07 | **64.01** |
| step | ~66 ms | **65.0 ms** |
| prefill peak | 3238 | 2941 |
| aggregate best | 87.9 | **123.0** |
| coherent | yes | yes |

Aggregate by concurrency: `{1: 34.6, 2: 62.4, 4: 78.9, 8: 123.0}`.

The earlier c8 figure of 82.2 was never a real measurement — with only 4 slots, 4
requests ran and 4 queued, so "c8" was really c4 plus a queue, and aggregate
*fell* below c4. With 8 slots throughput scales cleanly to c8 and aggregate rises
31% over the old c4 peak. Decode c1 improved too, which was not expected from a
concurrency knob; the likely cause is that the previous baseline shared the box
with queued work.

Prefill dropped ~9% (3238 -> 2941), the one regression. It is outweighed in the
composite score and prefill measurements carry more run-to-run noise than decode,
but it is worth re-checking if prefill later becomes the priority.

## 2026-08-19 — BLOCK_SIZE 16 -> 32 (ACCEPTED, hillclimb candidate 3)

Score 1.0801 -> 1.0884. A marginal win, recorded as such.

| metric | 16 | 32 |
|---|---|---|
| decode c1 median | 47.12 | **47.68** |
| prefill peak | 2941 | **3286** |
| aggregate best | 123.0 | 118.4 |
| coherent | yes | yes |

Prefill recovered the ~9% regression seen when MAX_NUM_SEQS went to 8 (+11.7%),
decode edged up 1.2%, aggregate fell 3.7%. Net +0.8% on the composite - inside the
range where run-to-run noise matters, so treat it as "not worse, probably slightly
better" rather than a clear win. BLOCK_SIZE=64 is next on the ladder.

Note: BLOCK_SIZE was not present in inkling-small.env at all - docker-compose
supplies it as ${BLOCK_SIZE:-16}. The first attempt at this candidate silently
changed nothing, because sed found no line to rewrite. It was caught only because
set_env now diffs the env file and asserts the intended key actually moved; without
that, the harness would have "measured" an unchanged config and recorded the noise
as a verdict on block size.

## 2026-08-19 — BLOCK_SIZE 32 REVERTED to 16 (scoring error, corrected)

The BLOCK_SIZE=32 acceptance logged above was made by a composite score that did
**not include peak decode**. Re-examined with peak included:

| | BLK=16 | BLK=32 |
|---|---|---|
| decode c1 median | 47.12 | 47.68 |
| decode c1 **peak** | **64.01** | 50.48 |
| aggregate | **123.0** | 118.4 |
| prefill | 2941 | **3286** |

BLOCK_SIZE=32 bought +1.2% median and +11.7% prefill at the cost of **27% of peak
decode and 3.9% of aggregate**. For a throughput-first objective that trade is
backwards, so 32 is reverted and **16 is the shipped value**.

The scoring function was re-weighted to match the actual goal:
aggregate 0.40 / median 0.25 / peak 0.25 / prefill 0.10. Under those weights
BLK=16 scores 0.9895 vs BLK=32 at 0.9352.

Lesson: an objective that omits a metric you care about will confidently optimise
it away.

## 2026-08-19 — Ladder exhausted: what is NOT tunable here

| candidate | result |
|---|---|
| `BLOCK_SIZE=64` | **crashes the engine**, 2/2 attempts |
| `KV_CACHE_DTYPE=fp8` | **fails to boot** on this SM121 stack |
| `KV_CACHE_MEMORY_BYTES=24GB` | OOM — engine capped at 106.5GB by `GPU_MEMORY_UTILIZATION=0.88`, and model+KV already use 103.6GB |
| `MAX_MODEL_LEN=327680` | needs ~21.6 GiB KV; only ~18.6 GiB fits |
| `MAX_NUM_BATCHED_TOKENS=16384` | no gain |

**262144 is the context ceiling for Inkling on this hardware.** KV costs a measured
69.2 KB/token; 1M tokens would need ~74 GB of KV on top of an ~83.6 GB model in a
121 GB box. Not reachable, with or without fp8.

## 2026-08-19 — DeepSeek-V4-Flash + DSpark comparison (same 2 Sparks)

Deployed `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` with gamma=5 DSpark parallel
drafting and `nvfp4_ds_mla` KV, to test whether the MTP3 ceiling is a model
limitation rather than a tuning one. It is.

| metric | Inkling (tuned) | DSpark V4-Flash |
|---|---|---|
| c1 decode median | 47.12 | **50.80** |
| c1 decode peak | **64.01** | 61.15 |
| aggregate peak | **123.0** (c8) | 102.8 (c4) |
| c1/c2/c4 aggregate | 34.6 / 62.4 / 78.9 | **45.6 / 72.9 / 102.8** |
| max context | 262,144 | **500,000** |
| acceptance ceiling | E<=4.0 (got 3.98) | **E<=6.0 (got 5.94)** |

DSpark reaches 98-100% acceptance at all five draft positions on predictable text,
which Inkling's serial MTP structurally cannot. Inkling retains the higher peak
aggregate only because the DSpark recipe caps `--max-num-seqs 6` and c6 already
regresses to 95.9 (saturated).

**Unreconciled:** the recipe README claims c1 decode 95.9 tok/s and c4 aggregate
263.7. Measured here: 50.80 and 102.8 — a 1.9-2.6x gap. Their figures are likely
warm-cache/thinking-mode/chat-endpoint; mine are `/v1/completions`, `ignore_eos`,
512-token windows over six diverse prompts, idle-gated. Not claiming their number
until it reproduces.
