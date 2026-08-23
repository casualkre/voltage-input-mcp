#!/usr/bin/env bash
# Start both llama-server instances for a profile, with role-specific tuning.
#
#   ./scripts/serve.sh [profile]   start (default: lean)
#   ./scripts/serve.sh --stop      stop
#   ./scripts/serve.sh --status    check
#
# Two separate servers rather than one, so each keeps its own KV cache and prompt-cache
# slot. A vision call can then never evict the actuator's cached prefix -- and that prefix
# is why the actuator costs ~90 ms per cycle instead of ~350 ms.
#
# The exact arguments come from llm/profiles.py so there is one source of truth. Run
# `voltage serve-models <profile>` to print them without starting anything.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
MODELS="${VOLTAGE_MODEL_DIR:-$ROOT/models}"
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/voltage-input-mcp"
SLOT_CACHE="$RUN_DIR/slot-cache"
PY="$ROOT/.venv/bin/python"
mkdir -p "$RUN_DIR" "$SLOT_CACHE"

find_server() {
  if [[ -x "$ROOT/vendor/llama.cpp/build/bin/llama-server" ]]; then
    echo "$ROOT/vendor/llama.cpp/build/bin/llama-server"
  elif command -v llama-server >/dev/null 2>&1; then
    command -v llama-server
  else return 1; fi
}

stop_all() {
  local stopped=0
  for pidfile in "$RUN_DIR"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "stopped $(basename "$pidfile" .pid) (pid $pid)"
      stopped=$((stopped+1))
    fi
    rm -f "$pidfile"
  done
  [[ $stopped -eq 0 ]] && echo "nothing was running"
  return 0
}

status() {
  for entry in "vision 8080" "actuator 8081"; do
    set -- $entry
    if curl -sf "http://127.0.0.1:$2/health" >/dev/null 2>&1; then
      echo "  $1   up    http://127.0.0.1:$2"
    else
      echo "  $1   down  http://127.0.0.1:$2"
    fi
  done
}

case "${1:-}" in
  --stop)   stop_all; exit 0 ;;
  --status) status; exit 0 ;;
esac

PROFILE="${1:-lean}"
SERVER=$(find_server) || {
  echo "error: llama-server not found." >&2
  echo "       Build it:  ./scripts/build-llama.sh" >&2
  echo "       Or use Ollama: set engine = \"ollama\" in voltage.toml" >&2
  exit 1
}

# GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 turns a VRAM overflow into a silent spill over PCIe:
# the server starts, works, and is roughly 10x slower with no error anywhere. Pin it off.
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=0

echo "profile $PROFILE"
echo "server  $SERVER"
if [[ ! -f "$ROOT/vendor/llama.cpp/build/bin/llama-server" ]]; then
  echo "note    using llama-server from PATH; if it was not built by build-llama.sh it may"
  echo "        lack GGML_CUDA_FA_ALL_QUANTS, in which case q8_0 KV + flash attention"
  echo "        falls back to a slow path. Check ./scripts/bench.sh output."
fi
echo

launch() {
  local role="$1"
  mapfile -t ARGS < <("$PY" - "$PROFILE" "$role" "$MODELS" "$SLOT_CACHE" <<'PY'
import sys
from voltage_input_mcp.llm.profiles import get_profile
profile, role, model_dir, slot_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
p = get_profile(profile)
spec = p.vision if role == "vision" else p.actuator
cpu = p.actuator_on_cpu and role == "actuator"
for arg in spec.llama_server_args(model_dir=model_dir, slot_cache_dir=slot_dir, cpu_only=cpu)[1:]:
    print(arg)
PY
) || return 1

  # Verify the weights are present before launching, so the failure is one clear line
  # rather than a wall of llama.cpp output.
  local i model
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--model" || "${ARGS[i]}" == "--mmproj" ]]; then
      model="${ARGS[i+1]}"
      [[ -s "$model" ]] || {
        echo "error: missing $(basename "$model")" >&2
        echo "       run: ./scripts/fetch-models.sh $PROFILE" >&2
        return 1
      }
    fi
  done

  echo "starting $role"
  printf '         %s\n' "$SERVER ${ARGS[*]}" | fold -sw 100 | sed '2,$s/^/         /'
  # setsid detaches into a new session, so the server survives this script's shell
  # exiting. A plain `&` leaves it in the caller's process group, which means it is
  # killed the moment the calling shell is torn down -- fine interactively, fatal when
  # serve.sh is invoked from another script, a CI job, or a tool runner.
  setsid nohup "$SERVER" "${ARGS[@]}" >"$RUN_DIR/$role.log" 2>&1 &
  echo $! > "$RUN_DIR/$role.pid"
  disown 2>/dev/null || true
}

launch vision   || exit 1
launch actuator || exit 1

echo
echo -n "waiting for both servers"
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1 \
  && curl -sf "http://127.0.0.1:8081/health" >/dev/null 2>&1; then
    echo " -- up."
    echo
    status
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null \
      | sed 's/^/  vram used\/free: /'
    echo
    echo "logs:   $RUN_DIR/{vision,actuator}.log"
    echo "bench:  ./.venv/bin/voltage bench"
    echo "stop:   ./scripts/serve.sh --stop"
    exit 0
  fi
  # Fail fast if a server died rather than waiting out the full timeout.
  for role in vision actuator; do
    pid=$(cat "$RUN_DIR/$role.pid" 2>/dev/null || echo)
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo " -- $role exited."
      echo
      tail -25 "$RUN_DIR/$role.log" >&2
      exit 1
    fi
  done
  echo -n "."
  sleep 1
done

echo " -- timed out."
echo "check: tail -40 $RUN_DIR/vision.log $RUN_DIR/actuator.log" >&2
exit 1
