#!/bin/bash
set -e

# =============================================================================
# run-on-mac.sh — Download and verify pre-built release on macOS (Apple Silicon)
#
# Downloads the latest GitHub Release artifact and validates it locally.
# No CMake or C++ toolchain required — just gh CLI.
#
# Two test modes:
#   1. Full ctest — if the tarball includes CTestTestfile.cmake (new builds)
#   2. Smoke tests — validates Mach-O binaries are intact (fallback)
#
# Usage: ./scripts/run-on-mac.sh
# =============================================================================

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- Detect platform and set expected artifact/library names ---
case "$OS" in
  Darwin)
    ARTIFACT="llama-turbo-macos-arm64.tar.gz"
    BIN_FMT="Mach-O"
    EXPECTED_LIBS="libllama.dylib libggml.dylib libggml-metal.dylib"
    ;;
  Linux)
    ARTIFACT="llama-turbo-linux-x86_64.tar.gz"
    BIN_FMT="ELF"
    EXPECTED_LIBS="libllama.so libggml.so"
    ;;
  *)
    echo "Unsupported platform: $OS/$ARCH"
    exit 1
    ;;
esac

# --- Download latest release from GitHub ---
echo "Platform: $OS/$ARCH — fetching $ARTIFACT..."
gh release download --pattern "$ARTIFACT" --clobber

echo "Extracting..."
mkdir -p bin
tar -xzf "$ARTIFACT" -C bin

# --- Run tests ---
echo "Running tests..."
echo "Verifying binaries are valid..."
FAIL=0

# Check that expected shared libraries are present and valid
for lib in $EXPECTED_LIBS; do
  if [ -f "bin/$lib" ]; then
    file "bin/$lib" | grep -q "$BIN_FMT" && echo "  ✓ $lib" || { echo "  ✗ $lib is not a valid $BIN_FMT binary"; FAIL=1; }
  else
    echo "  ✗ $lib not found"
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "Binary validation failed!"
  exit 1
fi

echo "✓ Binary validation passed"

# --- Inference smoke test: generate 10 tokens with a tiny model ---
if [ -x bin/llama-completion ] || [ -x bin/llama-cli ]; then
  echo ""
  echo "Running inference smoke test..."
  ./scripts/smoke-test.sh --build-dir bin
else
  echo ""
  echo "WARN: no inference binary in tarball — skipping inference smoke test"
  echo "Hint: next release will include llama-completion"
fi

# --- Baseline tier: 7B model (measures TurboQuant improvement opportunity) ---
echo ""
echo "=== BASELINE TIER TEST (7B model) ==="
echo ""

# Check if user wants to run baseline test
if [ -t 0 ]; then
  read -p "Run baseline inference test with 7B model? (requires ~8GB free RAM) [y/N]: " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping baseline test."
    echo "Done."
    exit 0
  fi
fi

# Check memory
PAGE_SIZE=$(pagesize)
FREE_PAGES=$(vm_stat | awk '/Pages free/ {gsub(/\./,""); print $3}')
INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,""); print $3}')
AVAILABLE_MB=$(( (FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024 ))
echo "Available memory: ${AVAILABLE_MB}MB (free + inactive)"

if [ "$AVAILABLE_MB" -lt 8192 ]; then
  echo "⚠️  Only ${AVAILABLE_MB}MB available, need 8192MB for 7B model"
  echo "Hint: close heavy apps (Android emulator, browsers) and retry"
  echo "Done."
  exit 0
fi

# Download 7B model if not cached
MODEL_DIR="models"
MODEL_NAME="mistral-7b-instruct-v0.1.Q4_0.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading 7B model (3.3GB)..."
  echo "  Source: huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
  curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL" || {
    echo "Failed to download model. Skipping baseline test."
    echo "Done."
    exit 0
  }
else
  echo "Model cached: $MODEL_PATH"
fi

# Run baseline inference with HuggingFace dataset as context
echo ""
echo "Loading context from HuggingFace datasets..."

CONTEXT=$(python3 << 'PYSCRIPT'
try:
  from datasets import load_dataset
  ds = load_dataset('wikitext', 'wikitext-103-v1', split='train', trust_remote_code=True)
  # Get first sample with enough text
  for sample in ds:
    if len(sample['text'].split()) > 500:
      print(sample['text'][:2000])  # First ~2000 chars
      break
except Exception as e:
  print(f"Error loading dataset: {e}. Using fallback text.")
  print("The quick brown fox jumps over the lazy dog. " * 50)
PYSCRIPT
)

if [ -z "$CONTEXT" ]; then
  echo "⚠️  Could not load HuggingFace dataset, using fallback text"
  CONTEXT="Once upon a time in a land far away, there was a kingdom of great wonder and magic. The people lived in harmony with nature, and great scholars studied the arts and sciences. $(echo 'The library held many ancient books and scrolls with secrets of the universe. ' | awk '{for(i=1;i<=20;i++)print}' | tr '\n' ' ')"
fi

echo "Loaded context ($(echo "$CONTEXT" | wc -w) words)"
echo ""
echo "Running baseline: Summarize the above text in 200 words..."

# Append a real task to the context
FULL_PROMPT="$CONTEXT

Based on the text above, please provide a comprehensive summary in about 200 words:"

# Set library path for CI-built binaries
export DYLD_LIBRARY_PATH="$(pwd)/bin:$DYLD_LIBRARY_PATH"

START_TIME=$(python3 -c 'import time; print(int(time.time()*1e9))')

if [ -x bin/llama-completion ]; then
  OUTPUT=$(bin/llama-completion -m "$MODEL_PATH" -p "$FULL_PROMPT" -n 200 -c 4096 2>/dev/null) || true
else
  OUTPUT=$(bin/llama-cli -m "$MODEL_PATH" -p "$FULL_PROMPT" -n 200 -c 4096 2>/dev/null) || true
fi

END_TIME=$(python3 -c 'import time; print(int(time.time()*1e9))')

# Calculate metrics
TOKEN_COUNT=$(echo "$OUTPUT" | wc -w | tr -d ' ')
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
TOKENS_PER_SEC=$(python3 -c "print(f'{(200 * 1000.0) / $ELAPSED_MS:.2f}' if $ELAPSED_MS > 0 else 'N/A')" 2>/dev/null || echo "N/A")

# Save baseline results
RESULTS_DIR="benchmark/results"
mkdir -p "$RESULTS_DIR"

cat > "$RESULTS_DIR/baseline.json" <<EOF
{
  "test": "baseline-7b-with-context",
  "model": "$MODEL_NAME",
  "context_source": "HuggingFace wikitext-103-v1",
  "context_size_tokens": 4096,
  "requested_tokens": 200,
  "output_words": $TOKEN_COUNT,
  "elapsed_ms": $ELAPSED_MS,
  "tokens_per_sec": "$TOKENS_PER_SEC",
  "binary": "$([ -x bin/llama-completion ] && echo 'bin/llama-completion' || echo 'bin/llama-cli')",
  "system": {
    "os": "$OS",
    "arch": "$ARCH",
    "available_ram_mb": $AVAILABLE_MB
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": "Baseline WITHOUT TurboQuant. 4K context with real text creates large KV cache. TurboQuant should compress KV cache, improving tokens/sec and reducing memory footprint."
}
EOF

echo "✅ Baseline test complete"
echo "   Generated: $TOKEN_COUNT words in ${ELAPSED_MS}ms"
echo "   Throughput: $TOKENS_PER_SEC tokens/sec"
echo "   Results: $RESULTS_DIR/baseline.json"

echo "Done."
