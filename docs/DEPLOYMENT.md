# Inkling-Small-NVFP4 on 2x DGX Spark — pinned deployment

Everything needed to reproduce the known-good serving config, with every version
pinned. Measured with this exact config on a clean box, 2026-08-18:

| metric | value |
|---|---|
| c1 decode | **43.0 tok/s median, 60.07 peak**, step 64.3-66.2ms |
| c4 aggregate | **94.1 tok/s** (best aggregate point) |
| max context | **262,144**, verified serving 234,221 prompt tokens |
| coherence | clean — 0 repeated 8-grams, natural EOS, on `/v1/chat/completions` |

## Files

| file | what it is |
|---|---|
| `Dockerfile` | reproducible build: pinned base digest, pinned patch commits, pinned pips, + the v37/v38 fixes as explicit layers |
| `Dockerfile.base` | the original build, unchanged, for reference/diffing |
| `docker-compose.yml` | the serving stack; `enable_flashinfer_autotune=True` is baked in |
| `.env.pinned` | every tuned value, each annotated with *why* and what breaks if changed |
| `patches/` | the two source patches applied by the Dockerfile |

## Pins

```
built image ID   sha256:d734208f04e1ca9b13ace21dacf0916945aacbb380e3d38124dbdf8825d6a6b3
base image       vllm/vllm-openai:nightly-65b7662d3fcb773afaf751ab29ac6960a0cf011d
                 @sha256:cec2df507519b5a2c4d9870524f86a2a100fae79cb689d1ce11f3d4da2f4ffad
vLLM commit      65b7662d3fcb773afaf751ab29ac6960a0cf011d
flash-attention  a80c4d7b5af32426002c42028a66a78408c273b2
MTP patch        0841fed141e9cb18e4464c2b4ec74881cdd90654
                 (sha256 fefb0dda787761106709110bf056d8ba42abfbf861e70b27e20a1d9f439dd206)
pip              av==18.0.0  scipy==1.18.0  soundfile==0.14.0  soxr==1.1.0
arch             TORCH_CUDA_ARCH_LIST=12.1a  FLASHINFER_CUDA_ARCH_LIST=12.1a  CUTE_DSL_ARCH=sm_121a
```

## Two ways to deploy

### A. Use the already-built image (strongest pin, no build)

The image already exists on both nodes and is pinned by ID above. Nothing to build:

```bash
cp .env.pinned inkling-small.env   # then edit the <head-ip>/<worker-ip> placeholders
docker image inspect local/inkling-vllm:v38-20260817 --format '{{.Id}}'
# must print sha256:d734208f04e1ca9b13ace21dacf0916945aacbb380e3d38124dbdf8825d6a6b3
bash start-managed.sh
```

### B. Rebuild from source

Requires the `image-root/` overlay tree — see below.

```bash
docker build -t local/inkling-vllm:v38-pinned .
```

## image-root/ — the one input not in this directory

`Dockerfile` does `COPY image-root/vllm/`. That tree is the SM121 paged-KV +
inkling model overlay. It is **not** in this directory — it lives at
`<deploy-dir>/image-root/` on the head node.

Copy it in before building:

```bash
scp -r <head>:<deploy-dir>/image-root ./
```

If the head is unavailable, extract it from the existing image instead:

```bash
cid=$(docker create local/inkling-vllm:v38-20260817)
mkdir -p image-root/vllm
docker cp "$cid":/usr/local/lib/python3.12/dist-packages/vllm/. image-root/vllm/
docker rm "$cid"
```

Note the extracted tree already has v37/v38 applied, so if you build from it,
comment out the two patch `RUN` lines or they will fail on already-patched files.

## Guardrails — these were learned by breaking things

**Change ONE variable per restart.** Stacking `MAX_MODEL_LEN` + `KV_CACHE_MEMORY_BYTES`
+ `MAX_NUM_SEQS` in a single restart overcommitted host RAM during engine init and
the OOM killer took out userspace — sshd included — requiring a physical power-cycle.

**`KV_CACHE_MEMORY_BYTES` floor is ~18.5GB.** At 16GB the engine refuses to boot:
`max seq len (262144) needs 17.3 GiB KV, larger than available 14.9 GiB`. 20GB is
proven. Lower it only together with `MAX_MODEL_LEN`.

**Clean stale rendezvous on BOTH nodes after any crash**, or NCCL fails with a
TCPStore broken-pipe on rank 1:

```bash
rm -f /dev/shm/psm_* /dev/shm/sem.mp-*
```

**Protect sshd from the OOM killer.** The box runs ~113/121 GB while serving and
sshd gets killed during model load, which costs you shell access on a live node:

```bash
sudo mkdir -p /etc/systemd/system/ssh.service.d
printf '[Service]\nOOMScoreAdjust=-1000\n' | sudo tee /etc/systemd/system/ssh.service.d/oom.conf
sudo systemctl daemon-reload
```

**Do not enable `inkling-small.service` autostart** while tuning — a bad config on
disk will re-wedge the box on every boot before you can log in to fix it.

## Do-not-change list (each measured, not assumed)

| setting | why |
|---|---|
| `MTP_NUM_TOKENS=3` | MTP4 raises acceptance but step time 69 -> 85-137ms. Net loss. |
| `MOE_BACKEND=flashinfer_cutlass` | trtllm won't boot on SM121; b12x/cutedsl JIT-storm and wedge the node; marlin only +1.5% |
| `enable_flashinfer_autotune=True` | worth ~7% step time. It is hard-coded in `docker-compose.yml`, NOT env-driven — grep for it if you regenerate the compose |
| `KV_CACHE_DTYPE=auto` | fp8 KV doubles context per GB but carries a decode cost on Spark; measure before adopting |
| NCCL on the MTU-9000 port | the second 200G port is idle (~8 MB vs 362 GB); binding the wrong one silently costs throughput |

## Benchmarking

Both harnesses read `INKLING_URL`, so they can be driven from the worker if the
head loses sshd:

```bash
INKLING_URL=http://<head>:8000 python3 bench_decode_v39.py speed 512 2
INKLING_URL=http://<head>:8000 python3 bench_decode_v39.py coherence
INKLING_URL=http://<head>:8000 python3 bench_conc_v39.py 1,2,4,8 384
INKLING_URL=http://<head>:8000 python3 probe_ctx_v39.py
```

`bench_decode_v39.py` settles the Prometheus spec counters and gates on an idle
engine. Both matter: without the settle, acceptance deltas smear across requests;
without the idle gate, other traffic on the endpoint inflates step time 68 -> 137ms.
Numbers taken without them are wrong.

## Known-open

- **c8 is unmeasured.** `MAX_NUM_SEQS=4` means 4 run + 4 queue, so the c8 figure
  (82.2 aggregate) is invalid. Set `MAX_NUM_SEQS=8` — alone — to measure it.
- **Context above 262144** needs KV, not model changes (`model_max_length` is
  1048576). 262144 costs 17.3 GiB, so ~26 GiB ≈ ~390k. Walk it up in steps.
- **65 tok/s c1 is not reachable at MTP3.** Ceiling is `4.0/step`; at 66.2ms that
  is 60.4 and we measure 60.07. Getting past it needs the draft path (19.55ms of
  66.91ms, ~85% unexplained by weights or launches) or a W4A16 MoE decode path.
