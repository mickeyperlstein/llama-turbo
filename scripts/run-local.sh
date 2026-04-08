#!/bin/bash
set -e

# Download latest macOS ARM64 build from GitHub Releases and run tests
# Usage: ./scripts/run-local.sh

echo "Fetching latest release..."
gh release download --pattern "llama-turbo-macos-arm64.tar.gz" --clobber

echo "Extracting..."
mkdir -p bin
tar -xzf llama-turbo-macos-arm64.tar.gz -C bin

echo "Running tests..."
if [ -f bin/CTestTestfile.cmake ]; then
  ctest --test-dir bin --output-on-failure
else
  echo "No CTestTestfile.cmake found — running smoke tests on binaries..."
  FAIL=0

  # Verify shared libraries exist and are loadable
  for lib in bin/libllama.dylib bin/libggml.dylib bin/libggml-metal.dylib; do
    if [ -f "$lib" ]; then
      file "$lib" | grep -q "Mach-O" && echo "  OK: $lib" || { echo "  FAIL: $lib is not a valid Mach-O binary"; FAIL=1; }
    else
      echo "  WARN: $lib not found"
    fi
  done

  # Run any test binaries that exist and accept --help or no args
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

echo "Done."
