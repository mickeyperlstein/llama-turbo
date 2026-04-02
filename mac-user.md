# Mac Setup Guide

This document explains what to do when setting up llama-turbo on your Apple Silicon Mac.
Written assuming you are working offline without the original architect available.

---

## What This Project Is

TurboQuant KV cache compression implemented as a `llama_memory_i` plugin for llama.cpp.
The goal is to compress the attention KV cache at runtime to 3.5–4.25 bits with no accuracy loss,
then submit the implementation upstream to `ggml-org/llama.cpp`.

All work is defined in `features/` as Gherkin `.feature` files. Read those to understand
what each sprint is trying to do. Do not modify files in `features/backlog/` or `features/closed/`.

---

## Quick Start

```bash
git clone https://github.com/mickeyperlstein/llama-turbo.git
cd llama-turbo
chmod +x scripts/setup-mac.sh
./scripts/setup-mac.sh
```

The script will:
1. Check and install prerequisites (brew, gh, cmake)
2. Authenticate GitHub CLI
3. Clone the `mickeyperlstein/llama.cpp` fork
4. Verify the llama.cpp Metal build works on your machine
5. Guide you through self-hosted runner registration (manual step, browser required)

---

## Self-Hosted Runner

The benchmark CI jobs run on your Mac, not GitHub's cloud runners.
This is required because benchmarks need the 32GB Apple Silicon unified memory.

After running the setup script, complete runner registration:

1. Go to: `https://github.com/mickeyperlstein/llama-turbo/settings/actions/runners/new`
2. Select macOS / ARM64
3. Follow GitHub's instructions — download the runner, configure it
4. When prompted for labels enter: `self-hosted,macos,apple-silicon,benchmark`
5. Start the runner: `./run.svc.sh start` (or as a background service)
6. Verify it shows as **Active** at: `https://github.com/mickeyperlstein/llama-turbo/settings/actions/runners`

---

## Running Tests Locally

To run tests without rebuilding from source, download the latest release artifact:

```bash
./scripts/run-local.sh
```

This downloads the pre-built macOS ARM64 binary from GitHub Releases and runs ctest.
No CMake or C++ toolchain needed.

---

## Repo Structure

```
features/
  backlog/   — sprints not yet started (read-only)
  current/   — active sprint (agents work here)
  closed/    — completed sprints with reports (read-only)

benchmark/
  results/   — all benchmark JSON output
  datasets/  — pinned dataset references

scripts/
  setup-mac.sh   — this setup script
  run-local.sh   — download and run latest build

.github/workflows/
  ci.yml              — Linux (Docker) + macOS CI on every push
  release.yml         — GitHub Release with build assets on merge to main
  sync-upstream.yml   — weekly PR to sync ggml-org/llama.cpp upstream
```

---

## Implementation Repo

All C++ work happens in the fork:
`https://github.com/mickeyperlstein/llama.cpp`

Do not commit TurboQuant implementation directly to `llama-turbo`.
`llama-turbo` is the coordination repo — feature definitions, benchmarks, CI only.

---

## Sprint Status

Check `features/current/` to see which sprint is active.
Check `features/closed/` and attached `*_report.md` files for completed work.
Check `features/backlog/` for upcoming sprints.

---

## If Something Is Broken

1. Check CI status: `https://github.com/mickeyperlstein/llama-turbo/actions`
2. Check open issues: `https://github.com/mickeyperlstein/llama-turbo/issues`
3. Read `CONTRIBUTING.md` for the full spec and rules
