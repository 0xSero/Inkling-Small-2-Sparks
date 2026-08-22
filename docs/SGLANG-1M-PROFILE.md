# The 1M-context profile (SGLang + DSpark + FP4 KV)

The vLLM profile in the repo root is the **throughput** champion (367.4 tok/s
aggregate). This is the second profile: **4× the context and the highest single-
stream peak measured on this hardware**, at the cost of batch throughput.

| | vLLM profile | this profile |
|---|---|---|
| max context | 262,144 | **1,048,576** |
| KV pool | 17.3 GiB @ 262,144 tok | **1,319,267 tokens** (`fp4_mx_block16`) |
| decode c1 peak | 64.0 | **85.9** |
| decode c1 median | 46.8 | 47.5 |
| aggregate | **367.4** @ c48 | ~109 @ c16 |
| verified | 234,221-token prompt | **307,581-token prompt, needle retrieved exactly** |

Pick this one for long documents and single-user latency. Pick vLLM for serving
many concurrent streams.

## What it is built from

Three pieces, none of them this repo's code:

1. **Base image** — public `lmsysorg/sglang`, pinned by digest
   `sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`.
2. **Patches + launchers** — [`0112358DEUS/inklingdeus`](https://github.com/0112358DEUS/inklingdeus).
   Its `scripts/bake-image.sh` overlays six patched SGLang files onto the base
   image, upgrades NCCL, and seeds `sm_121` MoE Triton configs by copying the
   `sm_100` ones. `KVQUANT=1` adds the FP4-KV overlay — that flag is what makes
   1M context fit.
3. **Draft head** — `RadixArk/Inkling-Small-DSpark-Preview` (1.7 GB, HF).
   This is a *trained* DSpark head, not a config-forced one; `block_size: 15`,
   `enable_confidence_head: true`, `dflash target_layer_ids [5,11,23,29,35]`.

An earlier attempt to run DSpark drafting under **vLLM** is a dead end: vLLM's
registry hardwires `"DSparkDraftModel" -> vllm.models.deepseek_v4`, which demands
`n_routed_experts` / `index_topk` / `hc_eps` / `hc_mult`. Forcing those fields
builds the wrong architecture and drafts garbage. The SGLang lane is the one
that actually loads this head.

## Bake the image (both nodes)

```bash
git clone https://github.com/0112358DEUS/inklingdeus ~/spark/inklingdeus
cd ~/spark/inklingdeus && KVQUANT=1 ./scripts/bake-image.sh
```

Produces `local/sglang-inkling:gb10` and `local/sglang-inkling:gb10-kvquant`.
Takes ~15 min and ~35 GB per node. If one node has no disk headroom, bake on the
node that does and stream it across the private link instead of baking twice:

```bash
docker save local/sglang-inkling:gb10-kvquant | ssh <worker> docker load
```

## Lay out the model directory (both nodes)

The launcher expects one directory containing **both** `inkling-small-nvfp4/`
and `dspark-draft/`. Build it with hardlinks so it costs no extra disk:

```bash
M=~/models/sglang-inkling
mkdir -p $M
cp -al ~/models/Inkling-Small-NVFP4 $M/inkling-small-nvfp4
SNAP=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Inkling-Small-DSpark-Preview/snapshots/*/ | head -1)
cp -aL $SNAP $M/dspark-draft      # -L: HF snapshots are symlinks into blobs/
```

`cp -al` (hardlink) for the weights, `cp -aL` (dereference) for the draft — the
HF cache stores snapshot files as symlinks into `blobs/`, and a bind-mounted
container cannot follow them out of the mount.

Expect `dspark-draft/` to hold `config.json  dflash.py  dspark.py  model.safetensors`.

## Launch — worker rank 1 FIRST, then head rank 0

```bash
# on BOTH nodes, same values except the rank argument
export MASTER_IP=<head IP on the private 200G link>
export IF=enp1s0f1np1          # NIC on that link
export HCA=rocep1s0f1          # its RDMA device   (ibv_devices)
export GID=3                   # RoCEv2 IPv4 GID   (show_gids | grep v2)
export MODELS=~/models/sglang-inkling
export BLOCK=10 MAXREQ=48
export GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16 24 32 40 48"

cd ~/spark/inklingdeus
./scripts/nvfp4-kv-boot.sh 1     # worker, FIRST
./scripts/nvfp4-kv-boot.sh 0     # head, after the worker is up
```

Serves OpenAI-compatible on **:30000** as model `inkling-small`. Cold boot is
~6–8 min (Triton/JIT warmup + CUDA graph capture at every `GRAPH_BS` size).

Rank order is not cosmetic: rank 1 blocks at `Init torch distributed begin` waiting
for the rendezvous. Start rank 0 first and it exits before the worker arrives.

Detach properly when launching over ssh — `ssh host 'nohup ... &'` without
`< /dev/null` and `ssh -f` dies with the ssh session mid-boot, leaving rank 1
hung at rendezvous with no rank 0 ever arriving.

Verify:

```bash
curl -s localhost:30000/health && grep max_total_num_tokens ~/inkling-serve.log | tail -1
# max_total_num_tokens=1319267 ... context_len=1048576
```

## Why BLOCK=10

`BLOCK` is the DSpark draft block size — how many tokens the draft head proposes
per verify. The head is trained to 15; the hardware optimum is lower. Measured,
3 reps, idle-gated:

| BLOCK | c1 peak | c1 median | verdict |
|---|---|---|---|
| 5 (upstream default) | 45.8 | 41.6 | baseline |
| 7 | 80.7 | 44.2 | big peak gain |
| **10** | **85.9** | **44.4 – 47.5** | **champion** |
| 15 | 84.2 | 35.6 | overdrafts — median collapses |

Peak keeps climbing to 10 because predictable text accepts nearly the whole
block. At 15 the draft spends real time producing tokens that get rejected on
ordinary prose, and the median — the number you actually feel — drops 20%.

**Large blocks are catastrophic for batch throughput.** At BLOCK=15 the aggregate
is pinned at ~55–61 tok/s at *any* concurrency (c16 54.3, c32 58.1, c48 60.8):
every stream is burning the GPU on speculative work that gets thrown away. This is
the whole reason the throughput profile stays on vLLM/MTP3 rather than moving here.

## Verified quality at depth

```bash
INKLING_URL=http://127.0.0.1:30000 PROBE_TOKENS=300000 python3 bench/probe_longctx.py
# prompt_tokens=307581 completion=57 wall=468.3s prefill~=657 tok/s
# finish=stop  FOUND_NEEDLE=True
# answer: TANGERINE-7741
```

A needle placed 35% into a 307,581-token document came back exactly, with a
natural stop — 45k tokens past the vLLM profile's hard ceiling. Prefill at that
depth is ~657 tok/s, so a 300k-token prompt costs ~8 minutes before first token.
That is the real cost of the long-context profile, and it is why it is a second
profile rather than the default.

Short-prompt coherence is clean too: all probes finish at their own EOS with zero
repeated 8-grams. Note this stack emits `reasoning_content` (100–700 words) that
the vLLM profile does not.

## Knobs deliberately left alone

- `CTX=65536` vs `1048576` — with FP4 KV the pool barely moves (1,104,683 vs
  1,082,627 tokens, ~2%), so 1M is essentially free. Take it.
- `PAGE=1` — page size 128 corrupts the triton verify path.
- `MOE=marlin` — the only numerically correct NVFP4 MoE runner on sm_121.
  (Note this is the *opposite* of the vLLM profile, where `flashinfer_cutlass`
  is the only working one. Different runtimes, different constraint.)
- `ATTN=triton` — `fa4` is sm_100-only.
- `MEMFRAC=0.85` — 0.87 boots but buys nothing measurable.
- `--kv-cache-dtype fp4_mx_block16` — *not* `nvfp4`. Same 0.5625 bytes/element,
  but `nvfp4` selects the flashinfer/trtllm recipe the triton lane cannot consume.
