#!/bin/bash
set -e

# =============================================================================
# run-on-linux.sh — Build, test, and package llama-turbo on Linux
#
# Called by both ci.yml and release.yml to keep build logic in one place.
# Two modes:
#   1. Inside a container (CI) — installs deps and builds directly
#   2. On a dev machine with Docker — builds via Dockerfile for reproducibility
#
# Metal is disabled — this is a CPU-only build for CI validation.
#
# Outputs:
#   - test-report.xml                    — ctest JUnit results
#   - llama-turbo-linux-x86_64.tar.gz    — packaged libs, test binaries, ctest config
#
# Usage: ./scripts/run-on-linux.sh
# =============================================================================

# --- Detect environment and install only what's missing ---
if command -v docker &>/dev/null && [ ! -f /.dockerenv ] && ! grep -q docker /proc/1/cgroup 2>/dev/null; then
  # Dev machine with Docker — delegate to Dockerfile for reproducibility
  echo "Docker detected — building via Dockerfile for reproducibility..."
  docker build -t llama-turbo .
  docker run --rm -v "$(pwd):/output" llama-turbo bash -c \
    'ctest --test-dir build --output-on-failure --output-junit /output/test-report.xml && \
     mkdir -p /tmp/staging && \
     cp -a build/bin/*.so* /tmp/staging/ 2>/dev/null || true && \
     for f in build/bin/test-*; do [ -x "$f" ] && cp "$f" /tmp/staging/; done && \
     cp build/CTestTestfile.cmake /tmp/staging/ 2>/dev/null || true && \
     cp -r build/tests /tmp/staging/tests 2>/dev/null || true && \
     tar -czf /output/llama-turbo-linux-x86_64.tar.gz -C /tmp/staging .'
  echo "Done."
  exit 0
fi

# --- CI runner or container — install only missing deps ---
# ubuntu-latest has cmake, clang, git, python3 pre-installed; just add pybind11
echo "Installing dependencies..."
sudo apt-get update && sudo apt-get install -y \
  pybind11-dev libpython3-dev 2>/dev/null \
  || apt-get update && apt-get install -y \
  build-essential cmake clang git \
  python3 python3-pip python3-dev \
  pybind11-dev libpython3-dev

# --- Configure cmake (no Metal on Linux) ---
echo "Configuring..."
cmake -B build \
  -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

# --- Build ---
echo "Building..."
cmake --build build --config Release -j"$(nproc)"

# --- Run tests and produce JUnit XML ---
echo "Running tests..."
ctest --test-dir build --output-on-failure --output-junit test-report.xml

# --- Package: libs + test binaries + ctest config into tarball ---
echo "Packaging artifacts..."
mkdir -p staging
cp -a build/bin/*.so* staging/ 2>/dev/null || true
for f in build/bin/test-*; do
  [ -x "$f" ] && cp "$f" staging/
done
cp build/CTestTestfile.cmake staging/ 2>/dev/null || true
cp -r build/tests staging/tests 2>/dev/null || true
tar -czf llama-turbo-linux-x86_64.tar.gz -C staging .

echo "Done."
