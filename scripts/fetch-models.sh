#!/usr/bin/env bash
# Download the GGUF weights for a profile, verifying each file exists on HuggingFace first.
#
# Usage:  ./scripts/fetch-models.sh [profile]     (default: lean)
#         ./scripts/fetch-models.sh --list

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
MODELS="${VOLTAGE_MODEL_DIR:-$ROOT/models}"
PROFILE="${1:-lean}"

if [[ "$PROFILE" == "--list" ]]; then
  exec ./.venv/bin/voltage profiles
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }

# Ask the package for the profile's file list rather than duplicating it here, so this
# script cannot drift from llm/profiles.py.
mapfile -t FILES < <(./.venv/bin/python - "$PROFILE" <<'PY'
import sys
from voltage_input_mcp.llm.profiles import get_profile
p = get_profile(sys.argv[1])
for spec in (p.vision, p.actuator):
    print(f"{spec.hf_repo}\t{spec.hf_file}")
    if spec.mmproj_repo and spec.mmproj_file:
        print(f"{spec.mmproj_repo}\t{spec.mmproj_file}")
PY
)

mkdir -p "$MODELS"
echo "profile: $PROFILE"
echo "target:  $MODELS"
echo

TOTAL=0
for entry in "${FILES[@]}"; do
  repo="${entry%%$'\t'*}"
  file="${entry##*$'\t'}"
  url="https://huggingface.co/$repo/resolve/main/$file"
  dest="$MODELS/$file"

  if [[ -s "$dest" ]]; then
    echo "  have    $file  ($(du -h "$dest" | cut -f1))"
    continue
  fi

  # HEAD first: a 404 here is a repo or filename that has moved, and it is much clearer
  # to say so than to leave a truncated .gguf behind.
  code=$(curl -sIL -o /dev/null -w '%{http_code}' "$url")
  if [[ "$code" != "200" ]]; then
    echo "  MISSING $file" >&2
    echo "          $url -> HTTP $code" >&2
    echo "          The repo or quant may have been renamed. Check:" >&2
    echo "          https://huggingface.co/$repo/tree/main" >&2
    exit 1
  fi

  echo "  fetch   $file"
  curl -L --fail --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  TOTAL=$((TOTAL + 1))
done

echo
echo "$TOTAL file(s) downloaded. Total on disk:"
du -sh "$MODELS"
echo
echo "next:  ./scripts/serve.sh $PROFILE"
