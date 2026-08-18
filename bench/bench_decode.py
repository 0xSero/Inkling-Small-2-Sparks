#!/usr/bin/env python3
"""Steady-state decode benchmark + acceptance audit for Inkling-Small on 2x DGX Spark.

Two modes:
  speed     - fixed-length decode with ignore_eos, to measure steady-state tok/s.
              The length here is a MEASUREMENT WINDOW, not a cap on what the model
              may say; coherence mode below runs uncapped.
  coherence - uncapped generation, judged by hand from the printed text.

Usage:
  python3 bench_decode_v39.py speed [n_tokens] [reps]
  python3 bench_decode_v39.py coherence
"""
import json
import sys
import time
import urllib.request

import os
BASE = os.environ.get("INKLING_URL", "http://127.0.0.1:8000")
MODEL = "inkling-small"

# Deliberately spans predictable -> diverse, since MTP acceptance is prompt-dependent
# and a single easy prompt overstates real throughput.
PROMPTS = [
    ("pattern", "Count: 1, 2, 3, 4,"),
    ("code", "Write a Python function that merges two sorted lists.\n\n"),
    ("explain", "Explain how a B-tree index speeds up database lookups.\n\n"),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\n"),
    ("list", "List the top 10 considerations when designing a distributed cache.\n\n"),
    ("reason", "A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph. When does the second catch the first? Think step by step.\n\n"),
]


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            return r.read().decode()
    except Exception:
        return ""


def running_reqs():
    """Requests currently executing on the engine (foreign traffic detector)."""
    for line in metrics().split("\n"):
        if line.startswith("vllm:num_requests_running"):
            try:
                return float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def wait_idle(timeout=300):
    """Block until the engine is idle. Foreign traffic (someone else using the
    endpoint) shares the GPU and inflates step time, so measuring through it
    produces silently wrong numbers."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if running_reqs() == 0:
            return True
        time.sleep(5)
    return False


def spec_counters():
    """Return (num_drafts, num_accepted, per_pos_accepted[]) from prometheus."""
    drafts = accepted = 0.0
    per_pos = {}
    for line in metrics().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        try:
            name, val = line.rsplit(" ", 1)
            v = float(val)
        except ValueError:
            continue
        if "spec_decode_num_draft_tokens" in name:
            drafts += v
        elif "spec_decode_num_accepted_tokens_total" in name:
            accepted += v
        elif "spec_decode_num_accepted_tokens_per_pos" in name and "position=" in name:
            pos = name.split('position="')[1].split('"')[0]
            per_pos[pos] = per_pos.get(pos, 0.0) + v
    return drafts, accepted, per_pos


def speed(n_tokens=512, reps=2):
    results = []
    print(f"=== decode speed: {n_tokens} tok window, {reps} reps, concurrency 1 ===\n")
    for name, prompt in PROMPTS:
        per_prompt = []
        for _ in range(reps):
            if not wait_idle():
                print(f"{name:9s} SKIPPED — engine busy with foreign traffic")
                continue
            # settle: vLLM flushes spec counters on an interval, so a delta taken
            # immediately around the request smears the previous request's tail in.
            time.sleep(2.0)
            d0, a0, p0 = spec_counters()
            t0 = time.perf_counter()
            resp = post("/v1/completions", {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": n_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            })
            dt = time.perf_counter() - t0
            # if foreign traffic joined mid-request we shared the GPU: mark it
            contended = running_reqs() > 0
            time.sleep(2.0)
            d1, a1, p1 = spec_counters()

            out_tok = resp["usage"]["completion_tokens"]
            toks = out_tok / dt
            dd = d1 - d0
            da = a1 - a0
            pos_rates = {}
            for k in sorted(set(p0) | set(p1), key=lambda x: int(x)):
                delta_pos = p1.get(k, 0) - p0.get(k, 0)
                # each position is drafted once per step; steps ~= dd / n_spec
                pos_rates[k] = delta_pos
            per_prompt.append((toks, dd, da, pos_rates, dt, out_tok, contended))
        if not per_prompt:
            continue
        best = max(x[0] for x in per_prompt)
        med = sorted(x[0] for x in per_prompt)[len(per_prompt) // 2]
        dd, da, pos, dt, out_tok, contended = per_prompt[-1][1:]
        steps = (dd / 3) if dd else 0
        E = (da / steps + 1) if steps else 0
        # step time from wall-clock and the number of decode steps actually taken
        step_ms = 1000 * dt / (out_tok / E) if E else 0
        pos_s = " ".join(f"p{k}={100*v/steps:.0f}%" for k, v in pos.items() if steps)
        results.append((name, best, med, E, step_ms))
        warn = "  !! CONTENDED" if contended else ""
        print(f"{name:9s} best={best:6.2f} med={med:6.2f} tok/s  E={E:.2f}  step={step_ms:5.1f}ms  {pos_s}{warn}")

    print()
    allbest = [r[1] for r in results]
    allmed = [r[2] for r in results]
    print(f"OVERALL  peak={max(allbest):.2f}  median-of-medians={sorted(allmed)[len(allmed)//2]:.2f}  worst={min(allmed):.2f}")
    return results


def coherence():
    """Uncapped-in-spirit generation on the CHAT endpoint (the real usage path).

    max_tokens is set high so the model stops at its own EOS rather than being
    truncated; the OpenAI default of 16 would otherwise cap every answer.
    """
    print("=== coherence: chat endpoint, natural stop ===\n")
    for name, prompt in PROMPTS[1:]:
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": 4096,
            "temperature": 0,
        })
        dt = time.perf_counter() - t0
        ch = resp["choices"][0]
        msg = ch.get("message", {})
        txt = msg.get("content") or ""
        think = msg.get("reasoning_content") or ""
        n = resp["usage"]["completion_tokens"]
        words = txt.split()
        grams = [" ".join(words[i:i+8]) for i in range(max(0, len(words) - 8))]
        dup = len(grams) - len(set(grams))
        flag = "  <-- REPETITION" if grams and dup > len(grams) * 0.15 else ""
        print(f"--- {name}  ({n} tok, {n/dt:.1f} tok/s, finish={ch.get('finish_reason')}, "
              f"think={len(think.split())}w, dup8={dup}){flag}")
        print(txt[:600].rstrip() or "(empty content)")
        print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "speed"
    if mode == "speed":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        r = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        speed(n, r)
    else:
        coherence()
