#!/bin/bash
set -e

echo "=== llama-turbo Mac Setup ==="
echo ""

# 1. Check prerequisites
echo "[1/5] Checking prerequisites..."
command -v brew >/dev/null 2>&1 || { echo "ERROR: Homebrew not found. Install from https://brew.sh"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found. Run: xcode-select --install"; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "Installing GitHub CLI..."; brew install gh; }
command -v cmake >/dev/null 2>&1 || { echo "Installing cmake..."; brew install cmake; }
echo "Prerequisites OK"
echo ""

# 2. Authenticate GitHub CLI
echo "[2/5] Checking GitHub auth..."
gh auth status >/dev/null 2>&1 || gh auth login
echo "GitHub auth OK"
echo ""

# 3. Clone llama.cpp fork
echo "[3/5] Cloning mickeyperlstein/llama.cpp fork..."
if [ ! -d "$HOME/llama.cpp" ]; then
    gh repo clone mickeyperlstein/llama.cpp "$HOME/llama.cpp" -- --depth=1
else
    echo "Already cloned at $HOME/llama.cpp — skipping"
fi
echo ""

# 4. Verify Metal build
echo "[4/5] Verifying llama.cpp builds with Metal..."
cd "$HOME/llama.cpp"
cmake -B build \
    -DLLAMA_METAL=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -Wno-dev 2>&1 | tail -5
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu) --target llama 2>&1 | tail -5
echo "Build OK"
echo ""

# 5. Register self-hosted runner
echo "[5/5] Self-hosted runner setup..."
echo ""
echo "  This step requires manual action in your browser."
echo ""
echo "  1. Go to: https://github.com/mickeyperlstein/llama-turbo/settings/actions/runners/new"
echo "  2. Select: macOS / ARM64"
echo "  3. Follow the instructions shown — copy and run each command in your terminal"
echo "  4. When asked for labels, add: self-hosted,macos,apple-silicon,benchmark"
echo "  5. When the runner shows as Active in GitHub, you're done"
echo ""
echo "=== Setup complete (runner registration pending) ==="
echo "See mac-user.md for full context."
