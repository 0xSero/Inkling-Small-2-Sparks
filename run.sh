#!/usr/bin/env bash
# Inkling-Small-NVFP4 on 2x DGX Spark — one-command bring-up.
#
#   ./run.sh              start serving on the fast config, then smoke-test it
#   ./run.sh chat "..."   one-shot prompt against the running server
#   ./run.sh bench        decode + concurrency + context benchmarks
#   ./run.sh stop         stop the stack on both nodes
#   ./run.sh status       health + live step-time decomposition
#
# The "fast config" is the measured optimum: MTP3 speculative decoding,
# flashinfer_cutlass MoE, NVFP4 sink experts, flashinfer autotune ON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${INKLING_ENV:-$HERE/deploy/inkling.env}"
COMPOSE="$HERE/deploy/docker-compose.yml"

die() { echo "error: $*" >&2; exit 1; }

bootstrap_env() {
  [ -f "$ENV_FILE" ] && return 0
  echo "no $ENV_FILE yet — creating from deploy/env.example"
  cp "$HERE/deploy/env.example" "$ENV_FILE"
  die "edit $ENV_FILE and set MASTER_ADDR / VLLM_HOST / VLLM_HOST_IP / WORKER_HOST / MODEL_ROOT, then re-run"
}

load_env() {
  bootstrap_env
  set -a; . "$ENV_FILE"; set +a
  case "${MASTER_ADDR:-}" in *"<"*) die "$ENV_FILE still has <placeholders> — fill them in";; esac
  URL="http://${MASTER_ADDR}:${VLLM_PORT:-8000}"
}

preflight() {
  command -v docker >/dev/null || die "docker not found"
  docker image inspect "$INKLING_VLLM_IMAGE" >/dev/null 2>&1 \
    || die "image $INKLING_VLLM_IMAGE not present — see docs/DEPLOYMENT.md (pull from GHCR or build)"
  # Stale NCCL rendezvous makes rank 1 die with a TCPStore broken pipe.
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
  [ -n "${WORKER_HOST:-}" ] && ssh -o ConnectTimeout=10 "$WORKER_HOST" \
      'rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null' 2>/dev/null || true
}

wait_healthy() {
  echo -n "waiting for engine (model load + graph capture is ~6-9 min) "
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1; then echo " ready"; return 0; fi
    echo -n "."; sleep 10
  done
  echo; die "engine did not come up — check: docker logs inkling-small-vllm-1"
}

smoke() {
  echo "--- smoke test ---"
  curl -sS --max-time 300 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"${SERVED_MODEL_NAME:-inkling-small}"'","messages":[{"role":"user","content":"In one sentence, what is speculative decoding?"}],"max_tokens":256,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
}

case "${1:-up}" in
  up)
    load_env; preflight
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d
    wait_healthy; smoke
    echo
    echo "serving at $URL   (OpenAI-compatible: $URL/v1/chat/completions)"
    ;;
  chat)
    load_env
    shift; PROMPT="${*:-Hello}"
    curl -sS --max-time 600 "$URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":4096,"temperature":0}))' "${SERVED_MODEL_NAME:-inkling-small}" "$PROMPT")" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
    ;;
  bench)
    load_env
    INKLING_URL="$URL" python3 "$HERE/bench/bench_decode.py" speed 512 2
    INKLING_URL="$URL" python3 "$HERE/bench/bench_concurrency.py" 1,2,4,8 384
    INKLING_URL="$URL" python3 "$HERE/bench/probe_context.py"
    ;;
  status)
    load_env
    curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1 && echo "health: OK" || echo "health: DOWN"
    docker exec inkling-small-vllm-1 sh -c 'tail -1 /tmp/decomp_v24.log' 2>/dev/null \
      || echo "(no decomp log — set INKLING_DECOMP=1)"
    ;;
  stop)
    load_env
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE" down || true
    [ -n "${WORKER_HOST:-}" ] && ssh -o ConnectTimeout=10 "$WORKER_HOST" \
        'docker rm -f inkling-small-vllm-1 2>/dev/null' 2>/dev/null || true
    echo stopped
    ;;
  *) die "usage: $0 {up|chat <prompt>|bench|status|stop}" ;;
esac
