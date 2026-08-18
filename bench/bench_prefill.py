#!/usr/bin/env python3
"""Prefill throughput for Inkling. Separates prefill from decode.

Prefill rate is measured as prompt_tokens / (wall - decode_time), where
decode_time is estimated from a 1-token run at the same prompt length. Asking for
max_tokens=1 still pays one decode step, so we subtract it rather than pretend it
is free.

Usage: python3 bench_prefill_v39.py [lengths_csv]
Reads INKLING_URL.
"""
import json, os, sys, time, urllib.request

BASE = os.environ.get("INKLING_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("INKLING_MODEL_NAME", "inkling-small")
UNIT = "The quick brown fox jumps over the lazy dog near the riverbank. "


def post(payload, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def running():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            for line in r.read().decode().split("\n"):
                if line.startswith("vllm:num_requests_running"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return 0.0


def wait_idle(timeout=300):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if running() == 0:
            return True
        time.sleep(5)
    return False


def measure(target):
    """Return (prompt_tokens, prefill_tok_s) or None."""
    prompt = UNIT * ((target // 12) + 8)
    if not wait_idle():
        return None
    # prefix caching would make the second call free, so vary the prompt slightly
    prompt = prompt + f" run={target}."
    t0 = time.perf_counter()
    r1 = post({"model": MODEL, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "ignore_eos": True})
    t_1 = time.perf_counter() - t0
    ptok = r1["usage"]["prompt_tokens"]

    # second call, same prompt, 9 more decode tokens -> isolates per-token decode
    # cost so it can be subtracted from the 1-token wall.
    if not wait_idle():
        return None
    t0 = time.perf_counter()
    post({"model": MODEL, "prompt": prompt, "max_tokens": 10,
          "temperature": 0, "ignore_eos": True})
    t_10 = time.perf_counter() - t0

    per_tok = max((t_10 - t_1) / 9.0, 0.0)
    prefill_s = max(t_1 - per_tok, 1e-6)
    return ptok, ptok / prefill_s, per_tok


def main():
    lens = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                             else ["1024", "4096", "16384", "65536"])]
    print(f"=== prefill ({BASE}) ===")
    out = []
    for L in lens:
        m = measure(L)
        if not m:
            print(f"{L:>7}: engine busy, skipped"); continue
        ptok, rate, per_tok = m
        out.append(rate)
        print(f"{L:>7} -> prompt_tok={ptok:>7}  prefill={rate:8.0f} tok/s  "
              f"(decode {1000*per_tok:5.1f} ms/tok)")
    if out:
        print(f"PREFILL_PEAK {max(out):.0f}")


if __name__ == "__main__":
    main()
