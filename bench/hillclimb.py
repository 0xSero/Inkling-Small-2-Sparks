#!/usr/bin/env python3
"""Hillclimb Inkling serving config: decode + prefill + concurrency, coherence-gated.

One candidate = ONE variable change. That rule is not stylistic: stacking
MAX_MODEL_LEN + KV_CACHE_MEMORY_BYTES + MAX_NUM_SEQS in a single restart
overcommitted host RAM during engine init, the OOM killer took userspace including
sshd, and the node needed a physical power-cycle. The safety checks below exist
because that already happened twice.

  python3 hillclimb.py step     # run exactly one candidate, accept or revert
  python3 hillclimb.py measure  # measure current config only, no changes
  python3 hillclimb.py status   # print state

State: hillclimb_state.json
Log:   hillclimb_history.jsonl
"""
import json, os, re, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "hillclimb_state.json"
HIST = HERE / "hillclimb_history.jsonl"

HEAD = os.environ.get("INKLING_HEAD", "head")
WORKER = os.environ.get("INKLING_WORKER", "worker")
DEPLOY = os.environ.get("INKLING_DEPLOY_DIR", "/opt/inkling-2-sparks")
ENVF = f"{DEPLOY}/inkling-small.env"

# --- hard safety limits ------------------------------------------------------
KV_FLOOR = 18_500_000_000        # below this the engine refuses to boot at 262144 ctx
MIN_FREE_GB_TO_RAISE_MEM = 12    # refuse memory-raising candidates without headroom
# KV bytes required per token of context, derived from the engine's own report
# (262144 ctx -> 17.3 GiB). Used to keep MAX_MODEL_LEN and KV consistent.
KV_BYTES_PER_TOKEN = 17.3 * 1024**3 / 262144

# --- candidate ladder --------------------------------------------------------
# Ordered by (expected gain / risk). Each entry changes exactly one key.
# "raises_mem" candidates get an extra free-RAM pre-flight.
CANDIDATES = [
    # concurrency headroom: c8 is currently unmeasurable (4 run + 4 queue)
    {"key": "MAX_NUM_SEQS", "value": "8", "raises_mem": True,
     "why": "make c8 a real measurement instead of 4 running + 4 queued"},
    # prefill throughput
    {"key": "MAX_NUM_BATCHED_TOKENS", "value": "16384", "raises_mem": True,
     "why": "larger prefill chunk -> better prefill tok/s"},
    # attention paging overhead
    {"key": "BLOCK_SIZE", "value": "32", "raises_mem": False,
     "why": "fewer page-table lookups; DSV4 reference recipe uses 256"},
    {"key": "BLOCK_SIZE", "value": "64", "raises_mem": False,
     "why": "continue the block-size ladder if 32 helped"},
    # context: KV-bound, walk up in steps, never jump
    {"key": "KV_CACHE_MEMORY_BYTES", "value": "24000000000", "raises_mem": True,
     "why": "headroom toward ~390k context; +4GB step only"},
    {"key": "MAX_MODEL_LEN", "value": "327680", "raises_mem": True,
     "why": "spend the KV raised in the previous step on context"},
    # decode: fp8 KV doubles context per GB but may cost decode on Spark
    {"key": "KV_CACHE_DTYPE", "value": "fp8", "raises_mem": False,
     "why": "2x context per GB; vLLM docs warn of a decode cost on Spark - measure it"},
    {"key": "GPU_MEMORY_UTILIZATION", "value": "0.90", "raises_mem": True,
     "why": "small headroom increase"},
]


def sh(cmd, timeout=120, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def head_ssh(cmd, timeout=120):
    return sh(f"ssh -o ConnectTimeout=15 {shlex.quote(HEAD)} {shlex.quote(cmd)}", timeout)


def head_url():
    addr = head_ssh(f"grep -m1 MASTER_ADDR {ENVF} | cut -d= -f2")
    port = head_ssh(f"grep -m1 VLLM_PORT {ENVF} | cut -d= -f2") or "8000"
    return f"http://{addr}:{port}"


def cluster_ready():
    """Both nodes reachable and the engine answering."""
    if "UP" not in sh(f"ssh -o ConnectTimeout=10 -o BatchMode=yes {HEAD} 'echo UP' 2>&1"):
        return False, "head unreachable"
    if "UP" not in sh(f"ssh -o ConnectTimeout=10 -o BatchMode=yes {WORKER} 'echo UP' 2>&1"):
        return False, "worker unreachable"
    ok = head_ssh("curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo OK || echo NO")
    return (ok == "OK"), ("engine healthy" if ok == "OK" else "engine not healthy")


def free_gb():
    out = head_ssh("free -g | sed -n 2p")
    try:
        return int(out.split()[6])       # 'available'
    except Exception:
        return 0


def read_env():
    raw = head_ssh(f"cat {ENVF}")
    env = {}
    for line in raw.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def set_env(key, value):
    head_ssh(f"cd {DEPLOY} && sed -i 's|^{key}=.*|{key}={value}|' inkling-small.env")


def validate(env, key, value):
    """Refuse configs known to brick the box or refuse to boot."""
    e = dict(env); e[key] = value
    kv = int(e.get("KV_CACHE_MEMORY_BYTES", "0") or 0)
    ctx = int(e.get("MAX_MODEL_LEN", "0") or 0)
    if kv < KV_FLOOR:
        return False, f"KV {kv} below floor {KV_FLOOR} - engine will refuse to boot"
    need = ctx * KV_BYTES_PER_TOKEN
    if e.get("KV_CACHE_DTYPE") == "fp8":
        need /= 2
    if need > kv * 0.95:
        return False, (f"ctx {ctx} needs ~{need/1024**3:.1f} GiB KV but only "
                       f"{kv/1024**3:.1f} GB allotted - raise KV first")
    return True, "ok"


def restart():
    head_ssh(f"cd {DEPLOY} && bash stop.sh >/dev/null 2>&1; true", timeout=180)
    time.sleep(12)
    for h in (HEAD, WORKER):
        sh(f"ssh -o ConnectTimeout=10 {h} 'rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null; true'")
    head_ssh(f"cd {DEPLOY} && nohup bash start-managed.sh > /tmp/hillclimb-boot.log 2>&1 & echo started")
    for i in range(90):
        time.sleep(20)
        ok = head_ssh("curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo OK || echo NO")
        if ok == "OK":
            return True
    return False


# --- measurement -------------------------------------------------------------
def measure():
    url = head_url()
    env = f"INKLING_URL={shlex.quote(url)}"
    res = {}

    dec = head_ssh(f"{env} python3 /tmp/bench_decode_v39.py speed 512 2", timeout=1800)
    res["decode_raw"] = dec
    m = re.search(r"peak=([\d.]+)\s+median-of-medians=([\d.]+)\s+worst=([\d.]+)", dec)
    if m:
        res["decode_peak"], res["decode_median"], res["decode_worst"] = (float(x) for x in m.groups())
    steps = [float(x) for x in re.findall(r"step=\s*([\d.]+)ms", dec)]
    res["step_ms"] = sum(steps) / len(steps) if steps else None
    res["contended"] = "CONTENDED" in dec

    pre = head_ssh(f"{env} python3 /tmp/bench_prefill_v39.py 1024,4096,16384,65536", timeout=1800)
    res["prefill_raw"] = pre
    m = re.search(r"PREFILL_PEAK ([\d.]+)", pre)
    res["prefill_peak"] = float(m.group(1)) if m else None

    con = head_ssh(f"{env} python3 /tmp/bench_conc_v39.py 1,2,4,8 384", timeout=2400)
    res["conc_raw"] = con
    aggs = {}
    for line in con.split("\n"):
        m = re.search(r"^c(\d+)\s+.*aggregate=\s*([\d.]+)", line.strip())
        if m:
            aggs[int(m.group(1))] = float(m.group(2))
    res["aggregate"] = aggs
    res["agg_best"] = max(aggs.values()) if aggs else None

    coh = head_ssh(f"{env} python3 /tmp/bench_decode_v39.py coherence", timeout=1800)
    res["coherence_raw"] = coh
    finishes = re.findall(r"finish=(\w+)", coh)
    dups = [int(x) for x in re.findall(r"dup8=(\d+)", coh)]
    res["coherent"] = bool(finishes) and all(f == "stop" for f in finishes) \
        and "REPETITION" not in coh and (not dups or max(dups) == 0)
    return res


def score(r):
    """Coherence is a hard gate; then weight the three axes the user asked for."""
    if not r.get("coherent"):
        return -1.0
    if r.get("contended"):
        return -1.0                      # contended runs are not comparable
    d = r.get("decode_median") or 0
    p = r.get("prefill_peak") or 0
    a = r.get("agg_best") or 0
    # normalised against the current known-good baseline so 1.0 == today
    return 0.5 * (d / 43.0) + 0.2 * (p / 3000.0) + 0.3 * (a / 94.1)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"best": None, "best_score": None, "tried": [], "next": 0}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def record(entry):
    with HIST.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def push_github(res, env, s):
    """Update the public repo only when a candidate genuinely wins."""
    repo = HERE / "deploy"
    readme = HERE.parent / "optimise-inkling" / "hillclimb_best.md"
    readme.write_text(
        "# Hillclimb — current best\n\n"
        f"score {s:.4f}\n\n"
        f"- decode c1 median **{res.get('decode_median')}** peak **{res.get('decode_peak')}** tok/s\n"
        f"- step **{res.get('step_ms')} ms**\n"
        f"- prefill peak **{res.get('prefill_peak')} tok/s**\n"
        f"- aggregate {res.get('aggregate')}\n"
        f"- coherent: {res.get('coherent')}\n\n"
        "```\n" + "\n".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n```\n")
    print(f"[github] wrote {readme} — commit/push handled by the loop driver")


def do_step():
    ok, why = cluster_ready()
    if not ok:
        print(f"SKIP: {why}")
        return 2

    st = load_state()
    env = read_env()

    if st["best_score"] is None:
        print("[baseline] measuring current config")
        r = measure()
        s = score(r)
        st["best"], st["best_score"] = env, s
        record({"t": time.time(), "kind": "baseline", "env": env, "res": r, "score": s})
        save_state(st)
        print(f"[baseline] score={s:.4f} decode_med={r.get('decode_median')} "
              f"prefill={r.get('prefill_peak')} agg={r.get('agg_best')} coherent={r.get('coherent')}")
        return 0

    while st["next"] < len(CANDIDATES):
        c = CANDIDATES[st["next"]]
        if env.get(c["key"]) == c["value"]:
            print(f"[skip] {c['key']}={c['value']} already set")
            st["next"] += 1; save_state(st); continue
        break
    else:
        print("[done] candidate ladder exhausted")
        return 1

    c = CANDIDATES[st["next"]]
    okv, msg = validate(env, c["key"], c["value"])
    if not okv:
        print(f"[reject-unsafe] {c['key']}={c['value']}: {msg}")
        st["next"] += 1; save_state(st); return 0
    if c["raises_mem"] and free_gb() < MIN_FREE_GB_TO_RAISE_MEM:
        print(f"[defer] {c['key']}: only {free_gb()}GB free, need {MIN_FREE_GB_TO_RAISE_MEM}")
        return 0

    prev = env.get(c["key"])
    print(f"[try] {c['key']}: {prev} -> {c['value']}  ({c['why']})")
    set_env(c["key"], c["value"])
    if not restart():
        print("[FAIL] did not come up — reverting")
        set_env(c["key"], prev); restart()
        record({"t": time.time(), "kind": "boot-fail", "cand": c})
        st["next"] += 1; save_state(st); return 0

    r = measure()
    s = score(r)
    base = st["best_score"]
    print(f"[result] score={s:.4f} vs best={base:.4f}  decode_med={r.get('decode_median')} "
          f"prefill={r.get('prefill_peak')} agg={r.get('agg_best')} coherent={r.get('coherent')}")
    record({"t": time.time(), "kind": "candidate", "cand": c, "res": r, "score": s})

    if s > base:
        print(f"[ACCEPT] {c['key']}={c['value']} improves {base:.4f} -> {s:.4f}")
        st["best"], st["best_score"] = read_env(), s
        push_github(r, st["best"], s)
    else:
        print(f"[REVERT] {c['key']} back to {prev}")
        set_env(c["key"], prev)
        restart()
    st["next"] += 1
    save_state(st)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "status":
        print(json.dumps(load_state(), indent=2))
    elif mode == "measure":
        ok, why = cluster_ready()
        print(why)
        if ok:
            r = measure(); print(json.dumps({k: v for k, v in r.items()
                                             if not k.endswith("_raw")}, indent=2))
            print("score:", score(r))
    else:
        sys.exit(do_step())
