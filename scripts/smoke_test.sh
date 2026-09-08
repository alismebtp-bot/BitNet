#!/usr/bin/env bash
# Smoke-test BitNet CPU inference against the official 2B GGUF.
# Usage:
#   ./scripts/smoke_test.sh [model.gguf]
# Exit 0 only if generation looks coherent (contains "Paris" for the France prompt).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf}"
PROMPT="The capital of France is"
EXPECT="Paris"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  echo "Download with:" >&2
  echo "  hf download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T" >&2
  exit 1
fi

if [[ ! -x build/bin/llama-cli && ! -x build/bin/Release/llama-cli.exe ]]; then
  echo "ERROR: llama-cli not built. Run: python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s" >&2
  exit 1
fi

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

python3 run_inference.py -m "$MODEL" -p "$PROMPT" -n 24 -t "${SMOKE_THREADS:-4}" -temp 0.0 \
  >"$OUT" 2>&1 || {
  echo "ERROR: inference failed. Last lines:" >&2
  tail -40 "$OUT" >&2
  exit 1
}

if ! grep -q "$EXPECT" "$OUT"; then
  echo "ERROR: expected '$EXPECT' in output. Got:" >&2
  tail -30 "$OUT" >&2
  exit 1
fi

echo "SMOKE OK: found '$EXPECT' in greedy completion"
grep -E "tokens per second|eval time|The capital of France" "$OUT" | tail -10 || true
