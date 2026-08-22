#!/usr/bin/env python3
"""Long-context probe: serve a ~300k-token prompt (past vLLM's 262,144 ceiling)
with a retrieval needle, verify the answer comes back correct and coherent."""
import json, time, urllib.request, os, random
BASE = os.environ.get("INKLING_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("INKLING_MODEL_NAME", "inkling-small")
random.seed(7)
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
filler_parts = []
n_target = int(os.environ.get("PROBE_TOKENS", "300000"))
# ~1.3 tok/word for this vocab -> ~ n_target words / 1.3
n_words = int(n_target / 1.3)
NEEDLE_POS = 0.35
needle = "The secret launch code is TANGERINE-7741."
for i in range(n_words // 12):
    filler_parts.append(" ".join(random.choice(words) for _ in range(12)) + ".")
insert_at = int(len(filler_parts) * NEEDLE_POS)
filler_parts.insert(insert_at, needle)
prompt = "Read this document carefully.\n\n" + "\n".join(filler_parts) + "\n\nQuestion: What is the secret launch code mentioned in the document? Answer with just the code."
t0 = time.perf_counter()
req = urllib.request.Request(BASE + "/v1/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": 32768, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    resp = json.loads(r.read().decode())
dt = time.perf_counter() - t0
u = resp["usage"]; ch = resp["choices"][0]
content = (ch.get("message") or {}).get("content") or ""
print(f"prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']} wall={dt:.1f}s prefill~={u['prompt_tokens']/dt:.0f} tok/s")
print(f"finish={ch.get('finish_reason')}  FOUND_NEEDLE={'TANGERINE-7741' in content}")
print("answer:", content[:200])
