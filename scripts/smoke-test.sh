#!/bin/bash
set -e

# =============================================================================
# smoke-test.sh — Run a 10-token inference smoke test with a tiny model
#
# Downloads stories15M-q4_0.gguf (~18MB) to models/ and generates 10 tokens
# using llama-completion. Verifies output is non-empty and writes results to
# benchmark/results/smoke.json.
#
# Requires: llama-completion (or llama-cli) binary in --build-dir or PATH
# Used by: run-on-mac.sh, benchmark.yml
#
# Usage: ./scripts/smoke-test.sh [--build-dir <path>] [--min-free-mb <MB>]
# =============================================================================

BUILD_DIR="bin"
MIN_FREE_MB=1024  # Minimum free RAM required (1GB for smoke, increase for larger models)

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Check available memory before running inference ---
echo "Checking system memory..."
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  PAGE_SIZE=$(pagesize)
  FREE_PAGES=$(vm_stat | awk '/Pages free/ {gsub(/\./,""); print $3}')
  INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,""); print $3}')
  AVAILABLE_MB=$(( (FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024 ))
  TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))

  echo "  Total RAM: ${TOTAL_MB}MB"
  echo "  Available: ${AVAILABLE_MB}MB (free + inactive)"
  echo "  Top memory consumers:"
  ps -eo pid,rss,comm | sort -k2 -rn | head -5 | while read pid rss comm; do
    echo "    $(( rss / 1024 ))MB  $comm"
  done
elif [ "$OS" = "Linux" ]; then
  AVAILABLE_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  TOTAL_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
  echo "  Total RAM: ${TOTAL_MB}MB"
  echo "  Available: ${AVAILABLE_MB}MB"
  echo "  Top memory consumers:"
  ps -eo pid,rss,comm --sort=-rss | head -6 | tail -5 | while read pid rss comm; do
    echo "    $(( rss / 1024 ))MB  $comm"
  done
fi

if [ "$AVAILABLE_MB" -lt "$MIN_FREE_MB" ] 2>/dev/null; then
  echo ""
  echo "FAIL: Only ${AVAILABLE_MB}MB available, need ${MIN_FREE_MB}MB"
  echo "Hint: close heavy apps (browsers, IDEs, Docker) and retry"
  exit 1
fi
echo "  Memory check passed (${AVAILABLE_MB}MB >= ${MIN_FREE_MB}MB required)"
echo ""

# --- Locate inference binary (prefer llama-completion, fall back to llama-cli) ---
if [ -x "$BUILD_DIR/llama-completion" ]; then
  LLAMA_BIN="$BUILD_DIR/llama-completion"
elif [ -x "$BUILD_DIR/llama-cli" ]; then
  LLAMA_BIN="$BUILD_DIR/llama-cli"
elif command -v llama-completion &>/dev/null; then
  LLAMA_BIN="llama-completion"
elif command -v llama-cli &>/dev/null; then
  LLAMA_BIN="llama-cli"
else
  echo "FAIL: neither llama-completion nor llama-cli found in $BUILD_DIR/ or PATH"
  echo "Hint: ensure the release tarball includes llama-completion, or build from source"
  exit 1
fi
echo "Using: $LLAMA_BIN"

# --- Download tiny test model if not cached ---
MODEL_DIR="models"
MODEL_NAME="stories15M-q4_0.gguf"
MODEL_SIZE="18MB"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading model: $MODEL_NAME ($MODEL_SIZE) -> $MODEL_PATH"
  echo "  Source: huggingface.co/ggml-org/models"
  curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
  echo "  Download complete."
else
  echo "Model cached: $MODEL_PATH ($MODEL_SIZE)"
fi
echo ""

# --- Run 10-token smoke generation ---
echo "Running 10-token smoke generation..."
PROMPT="Once upon a time"

START_TIME=$(python3 -c 'import time; print(int(time.time()*1e9))')

OUTPUT=$("$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  -p "$PROMPT" \
  -n 10 \
  -c 128 \
  2>/dev/null) || true

END_TIME=$(python3 -c 'import time; print(int(time.time()*1e9))')

# --- Validate output ---
if [ -z "$OUTPUT" ]; then
  echo "FAIL: smoke generation produced no output"
  exit 1
fi

TOKEN_COUNT=$(echo "$OUTPUT" | wc -w | tr -d ' ')
echo "  Output: $OUTPUT"
echo "  Words: $TOKEN_COUNT"

# --- Calculate timing ---
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
echo "  Time: ${ELAPSED_MS}ms"
echo ""

# --- Write smoke.json ---
RESULTS_DIR="benchmark/results"
mkdir -p "$RESULTS_DIR"

# Escape output for JSON
OUTPUT_ESCAPED=$(echo "$OUTPUT" | head -1 | sed 's/"/\\"/g' | tr -d '\n')

cat > "$RESULTS_DIR/smoke.json" <<ENDJSON
{
  "test": "smoke-generation",
  "model": "$MODEL_NAME",
  "prompt": "$PROMPT",
  "requested_tokens": 10,
  "output_words": $TOKEN_COUNT,
  "output_preview": "$OUTPUT_ESCAPED",
  "elapsed_ms": $ELAPSED_MS,
  "binary": "$LLAMA_BIN",
  "system": {
    "os": "$OS",
    "arch": "$(uname -m)",
    "total_ram_mb": $TOTAL_MB,
    "available_ram_mb": $AVAILABLE_MB
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "passed": true
}
ENDJSON

echo "Smoke test passed. Results written to $RESULTS_DIR/smoke.json"
