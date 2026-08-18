#!/usr/bin/env python3
"""Probe the largest context the live deployment actually serves, and what decode
costs at that length. Reads INKLING_URL from env."""
import json, os, sys, time, urllib.request

BASE = os.environ.get("INKLING_URL", "http://127.0.0.1:8000")


def post(payload, timeout=1800):
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    info = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=20).read())
    print("served max_model_len:", info["data"][0].get("max_model_len"))

    unit = "The quick brown fox jumps over the lazy dog near the riverbank. "
    targets = [int(x) for x in (sys.argv[1:] or ["16384", "65536", "131072", "200000", "250000"])]
    for target in targets:
        prompt = unit * ((target // 12) + 64)
        try:
            t0 = time.perf_counter()
            r = post({"model": "inkling-small", "prompt": prompt,
                      "max_tokens": 64, "temperature": 0, "ignore_eos": True})
            dt = time.perf_counter() - t0
            u = r["usage"]
            print("target~{:>7} -> prompt_tok={:>7}  decode={:6.1f} tok/s  wall={:6.1f}s  OK".format(
                target, u["prompt_tokens"], u["completion_tokens"] / dt, dt))
        except Exception as e:
            print("target~{:>7} -> FAILED: {}".format(target, str(e)[:160]))


if __name__ == "__main__":
    main()
