#!/usr/bin/env python3
"""Concurrency sweep for Inkling: per-stream and aggregate decode tok/s at c1/2/4/8.

Reports BOTH numbers because they answer different questions:
  per-stream  - what one user feels (this is the "c1 tok/s" metric)
  aggregate   - total tokens/s the box emits across all streams

Usage: python3 bench_conc_v39.py [levels] [n_tokens]
       python3 bench_conc_v39.py 1,2,4,8 384
"""
import json, sys, time, threading, urllib.request

import os
BASE = os.environ.get("INKLING_URL", "http://127.0.0.1:8000")
MODEL = "inkling-small"

PROMPTS = [
    "Explain how a B-tree index speeds up database lookups.\n\n",
    "Write a Python function that merges two sorted lists.\n\n",
    "List the top 10 considerations when designing a distributed cache.\n\n",
    "Write a short story about a lighthouse keeper.\n\n",
    "Summarise why vectorised execution helps analytical queries.\n\n",
    "Describe the tradeoffs between LRU and LFU eviction.\n\n",
    "What makes a good incident postmortem? Be specific.\n\n",
    "Explain consistent hashing to a new engineer.\n\n",
]


def post(payload, timeout=900):
    req = urllib.request.Request(BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics_running():
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
        if metrics_running() == 0:
            return True
        time.sleep(5)
    return False


def run_level(conc, n_tokens):
    if not wait_idle():
        print(f"c{conc}: engine busy, skipped"); return
    results = [None] * conc
    def worker(i):
        t0 = time.perf_counter()
        r = post({"model": MODEL, "prompt": PROMPTS[i % len(PROMPTS)],
                  "max_tokens": n_tokens, "temperature": 0,
                  "ignore_eos": True, "stream": False})
        dt = time.perf_counter() - t0
        results[i] = (r["usage"]["completion_tokens"], dt)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0

    ok = [x for x in results if x]
    total_tok = sum(t for t, _ in ok)
    per_stream = [t / d for t, d in ok]
    agg = total_tok / wall
    print(f"c{conc:<2} per-stream: min={min(per_stream):5.1f} "
          f"med={sorted(per_stream)[len(per_stream)//2]:5.1f} "
          f"max={max(per_stream):5.1f} tok/s   aggregate={agg:6.1f} tok/s   "
          f"wall={wall:5.1f}s")


if __name__ == "__main__":
    levels = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8").split(",")]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    print(f"=== concurrency sweep, {n} tok/stream ===")
    for c in levels:
        run_level(c, n)
