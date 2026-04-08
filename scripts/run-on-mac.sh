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
if [ -f bin/CTestTestfile.cmake ]; then
  # Full ctest suite — available when tarball includes cmake test config
  ctest --test-dir bin --output-on-failure
else
  # Smoke tests — verify binaries are valid without a full test harness
  echo "No CTestTestfile.cmake found — running smoke tests on binaries..."
  FAIL=0

  # Check that expected shared libraries are present and valid
  for lib in $EXPECTED_LIBS; do
    if [ -f "bin/$lib" ]; then
      file "bin/$lib" | grep -q "$BIN_FMT" && echo "  OK: $lib" || { echo "  FAIL: $lib is not a valid $BIN_FMT binary"; FAIL=1; }
    else
      echo "  WARN: $lib not found"
    fi
  done

  # Execute any packaged test binaries
  for test_bin in bin/test-*; do
    if [ -x "$test_bin" ]; then
      echo "  Running: $test_bin ..."
      if "$test_bin" 2>&1 | head -5; then
        echo "  OK: $test_bin"
      else
        echo "  FAIL: $test_bin exited with error"
        FAIL=1
      fi
    fi
  done

  if [ "$FAIL" -ne 0 ]; then
    echo "Some smoke tests failed!"
    exit 1
  fi

  echo "Smoke tests passed."
fi

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

echo "Done."
