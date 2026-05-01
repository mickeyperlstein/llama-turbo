#!/usr/bin/env bash
# dev.sh — developer workflow script for llama-turbo
# Claude edits TASK= and ARGS= below, you run this file once with permission.
#
# Usage: ./scripts/dev.sh
# To grant permission: in Claude Code settings, allow execution of this file.
#
# TASK controls what action to take:
#   build         — cmake configure + build changed targets
#   rebuild       — full clean cmake + build
#   test-stories  — bench on stories15M with TBQ4_0
#   test-mistral  — bench on Mistral-7B with TBQ4_0 (requires model)
#   bench-baseline — full baseline bench (F16 KV, no turboq)
#   smoke         — run smoke-test.sh against current build
#   run-on-mac    — run the official release test script
#   custom        — run CUSTOM_CMD directly
#   run-multi     — run multiple commands from RUN_MULTI_CMDS
#
# Claude sets these:
TASK="custom"
BUILD_DIR="build_turboq"
ARGS=""
CUSTOM_CMD='cd build_turboq_ext && ../.venv/bin/python3 -c "
import sys; sys.path.insert(0, \".\")
import turboq_ext, numpy as np

# Debug 3-bit: use all-zeros input (norm=0 path)
z = np.zeros(256, dtype=np.float16)
enc_z = turboq_ext.encode(z.view(np.uint16), 0, 0, 3)
dec_z = turboq_ext.decode(enc_z, 256, 0, 0, 3).view(np.float16)
print(f\"zero vec 3-bit: max_err={float(np.max(np.abs(dec_z.astype(np.float32)))):.4f}\")

# Debug 3-bit: check first few decoded values vs encoded indices
rng = np.random.default_rng(42)
v = rng.standard_normal(256).astype(np.float16)
enc3 = turboq_ext.encode(v.view(np.uint16), 0, 0, 3)
dec3 = turboq_ext.decode(enc3, 256, 0, 0, 3).view(np.float16)
print(f\"3-bit bytes={len(enc3)}, expected=98\")

# Check if norm round-trips correctly through f16
# block_tbq3_0 layout: d (fp16, 2 bytes) at offset 0, then qs[96]
norm_approx = float(np.sqrt(np.sum(v.astype(np.float32)**2)))
enc3_bytes = bytearray(enc3)
norm_f16 = int.from_bytes(enc3_bytes[0:2], byteorder=\"little\")
# Convert f16 to f32 manually
s = (norm_f16 >> 15) & 1
e = (norm_f16 >> 10) & 0x1f
m = norm_f16 & 0x3ff
import struct
if e == 0: f32_exp = 1
else: f32_exp = e
bits32 = (s << 31) | ((f32_exp + 112) << 23) | (m << 13)
norm_stored = struct.unpack(\"f\", struct.pack(\"I\", bits32))[0]
print(f\"norm actual={norm_approx:.4f}, stored={norm_stored:.4f}\")

# Check the actual RMSE
rmse3 = float(np.sqrt(np.mean((v.astype(np.float32) - dec3.astype(np.float32))**2)))
print(f\"3-bit RMSE: {rmse3:.4f}\")
print(f\"dec3 min={float(np.min(dec3.astype(np.float32))):.3f} max={float(np.max(dec3.astype(np.float32))):.3f}\")
print(f\"v   min={float(np.min(v.astype(np.float32))):.3f} max={float(np.max(v.astype(np.float32))):.3f}\")

# Also test 4-bit for comparison
enc4 = turboq_ext.encode(v.view(np.uint16), 0, 0, 4)
dec4 = turboq_ext.decode(enc4, 256, 0, 0, 4).view(np.float16)
rmse4 = float(np.sqrt(np.mean((v.astype(np.float32) - dec4.astype(np.float32))**2)))
print(f\"4-bit RMSE: {rmse4:.4f}\")
"'

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
DYLD_LIBRARY_PATH="$ROOT/$BUILD_DIR/bin${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH

case "$TASK" in
  build)
    cmake --build "$BUILD_DIR" -j"$NPROC" $ARGS
    ;;

  rebuild)
    rm -rf "$BUILD_DIR"
    cmake -B "$BUILD_DIR" -S llama.cpp \
      -DGGML_METAL=ON \
      -DLLAMA_TURBOQ=ON \
      -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DTURBOQ_SRC_DIR="$ROOT/turboq" \
      $ARGS
    cmake --build "$BUILD_DIR" -j"$NPROC"
    ;;

  build-target)
    # Build specific targets (set ARGS="llama llama-bench" etc)
    cmake --build "$BUILD_DIR" --target $ARGS -j"$NPROC"
    ;;

  test-stories)
    MODEL="models/stories15M-q4_0.gguf"
    if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
    "$BUILD_DIR/bin/llama-bench" \
      -m "$MODEL" \
      -ctk tbq4_0 -ctv f16 \
      -fa 1 -nkvo 1 \
      -p 32 -n 16 -r 2 $ARGS 2>&1
    ;;

  test-mistral)
    MODEL="models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
    "$BUILD_DIR/bin/llama-bench" \
      -m "$MODEL" \
      -ctk tbq4_0 -ctv f16 \
      -fa 1 -nkvo 1 \
      -p 64 -n 32 -r 2 $ARGS 2>&1
    ;;

  bench-baseline)
    MODEL="models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
    "$BUILD_DIR/bin/llama-bench" \
      -m "$MODEL" \
      -ctk f16 -ctv f16 \
      -p 64 -n 32 -r 2 $ARGS 2>&1
    ;;

  bench-compare)
    # Side-by-side: F16 vs TBQ4_0 on Mistral
    MODEL="models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
    echo "=== F16 KV cache (baseline) ==="
    "$BUILD_DIR/bin/llama-bench" -m "$MODEL" -ctk f16  -ctv f16 -p 64 -n 32 -r 2 2>&1 | grep -v "^ggml_metal"
    echo ""
    echo "=== TBQ4_0 KV cache (turboq) ==="
    "$BUILD_DIR/bin/llama-bench" -m "$MODEL" -ctk tbq4_0 -ctv f16 -fa 1 -nkvo 1 -p 64 -n 32 -r 2 2>&1 | grep -v "^ggml_metal"
    ;;

  build-ext)
    PYEXE="$ROOT/.venv/bin/python3"
    cmake -B build_turboq_ext -S turboq -DPython3_EXECUTABLE="$PYEXE" -DCMAKE_BUILD_TYPE=Release
    cmake --build build_turboq_ext -j"$NPROC"
    echo "Extension built: build_turboq_ext/turboq_ext*.so"
    ;;

  smoke)
    ./scripts/smoke-test.sh --build-dir "$BUILD_DIR/bin"
    ;;

  run-on-mac)
    ./scripts/run-on-mac.sh
    ;;

  debug-mistral)
    MODEL="models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
    lldb --batch \
      -o "env DYLD_LIBRARY_PATH=$ROOT/$BUILD_DIR/bin" \
      -o "run" \
      -o "bt" \
      -o "exit" \
      -- "$BUILD_DIR/bin/llama-bench" \
         -m "$MODEL" -ctk tbq4_0 -ctv f16 -fa 1 -nkvo 1 -p 64 -n 32 -r 1 2>&1
    ;;

  custom)
    if [ -z "$CUSTOM_CMD" ]; then echo "CUSTOM_CMD is empty"; exit 1; fi
    eval "$CUSTOM_CMD"
    ;;

  run-multi)
    if [ -z "$RUN_MULTI_CMDS" ]; then
      echo "RUN_MULTI_CMDS is empty"
      exit 1
    fi

    echo "Running multiple commands:"
    while IFS= read -r line; do
      # Skip empty lines
      if [ -z "$line" ]; then
        continue
      fi

      echo "run: $line"
      eval "$line"
      status=$?

      if [ $status -ne 0 ]; then
        echo "run: Command failed with exit code $status"
        exit $status
      fi
    done <<< "$RUN_MULTI_CMDS"
    ;;

  *)
    echo "Unknown TASK: $TASK"
    echo "Valid tasks: build rebuild build-target test-stories test-mistral bench-baseline bench-compare smoke run-on-mac custom"
    exit 1
    ;;

esac
