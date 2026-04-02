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
ctest --test-dir bin --output-on-failure

echo "Done."
