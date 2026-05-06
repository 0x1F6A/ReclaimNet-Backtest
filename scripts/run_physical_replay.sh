#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <build_dir> <replay.json> <new_run_dir>" >&2
  echo "Requires real Linux GPU hosts with CUDA, NVML, and the ReclaimNet services installed." >&2
  exit 1
fi

for key in RNBT_FAKE_GPU_DIR RNBT_FAKE_GPUS RNBT_FAKE_VRAM_BYTES; do
  if [ -n "${!key:-}" ]; then
    echo "$key is unsupported; this runner requires real GPUs." >&2
    exit 1
  fi
done

BUILD=$(cd "$1" && pwd)
CONFIG=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
RUN=$(cd "$3" && pwd)
BIN=$BUILD/bin

if [ -e "$RUN/run_manifest.json" ]; then
  echo "Refusing to overwrite an existing run: $RUN" >&2
  exit 1
fi
for tool in rnbt-replay rnbt-evaluate; do
  if [ ! -x "$BIN/$tool" ]; then
    echo "Missing executable: $BIN/$tool" >&2
    exit 1
  fi
done

"$BIN/rnbt-replay" plan "$CONFIG" --set "/output=$RUN" --out "$RUN/plan.json"

cleanup_needed=1
cleanup() {
  if [ "$cleanup_needed" -eq 1 ]; then
    "$BIN/rnbt-replay" testbed "$CONFIG" stop > "$RUN/testbed_stop.json" 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

status=0
"$BIN/rnbt-replay" run "$CONFIG" --set "/output=$RUN" || status=$?
if "$BIN/rnbt-replay" testbed "$CONFIG" stop > "$RUN/testbed_stop.json" 2>&1; then
  cleanup_needed=0
else
  if [ "$status" -eq 0 ]; then status=1; fi
fi
if ! "$BIN/rnbt-replay" collect "$CONFIG" --run "$RUN"; then
  if [ "$status" -eq 0 ]; then status=1; fi
fi
if [ "$status" -eq 0 ]; then
  "$BIN/rnbt-evaluate" "$RUN" || status=$?
fi

echo "Run artifacts: $RUN"
exit "$status"
