# llama-turbo

**TurboQuant: KV Cache Compression for Fast Inference**

Optimizing large language model inference through learned KV cache quantization. Reducing memory bandwidth bottlenecks in transformer attention without sacrificing output quality.

Architected and built by Mickey Perlstein using Claude Code agents.
---

## Overview

llama-turbo uses an interface called kv_cache and we use it to swap out the default kv cache implementation with our own quantized version. This allows us to implement **KV cache quantization** to accelerate inference on large language models. Instead of storing full-precision key-value pairs in attention, we learn to compress them dynamically, reducing memory pressure during token generation.

**Baseline Performance (Sprint 0):** 7.05 tokens/sec on Mistral-7B with 8K context window  
**Target (Sprint 1+):** 2-3x improvement via KV cache compression

---

## Key References

- **KV Cache Quantization:** [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research: 6x KV cache compression with PolarQuant + QJL transform
- **Base Framework:** [llama.cpp](https://github.com/ggerganov/llama.cpp) — Efficient inference in C++
- **Model Distribution:** [Ollama](https://ollama.ai) — Simple LLM deployment
- **Model Hub:** [Hugging Face](https://huggingface.co) — Model & dataset repository

---

## Models & Data

**Baseline Models:**
- [Mistral-7B-Instruct-v0.1 (Q4_0)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) — 7B parameters, 4-bit quantization
- [stories15M-q4_0](https://huggingface.co/ggml-org/models) — 15M parameter tiny model for smoke tests

**Datasets:**
- [WikiText-103-v1](https://huggingface.co/datasets/wikitext/tree/main/data) — Real-world corpus for baseline stress tests (1810 words context)

---

## Project Structure

```
llama-turbo/
├── llama.cpp/              # Submodule: upstream inference engine
├── scripts/
│   ├── run-on-mac.sh       # Local macOS test harness (download, validate, benchmark)
│   ├── smoke-test.sh       # Lightweight inference validation
│   └── verify-release.sh   # Check release artifacts
├── .github/workflows/
│   ├── ci.yml              # Linux + macOS builds on every push/PR
│   ├── benchmark.yml       # Inference smoke tests on main
│   └── release.yml         # Package artifacts + benchmark results
├── benchmark/results/      # Baseline measurements (JSON)
├── tests/                  # Upstream llama.cpp tests
└── SPRINT_0_REPORT.md      # Sprint 0 completion report
```

---

## Development Sprints

| Sprint | Goal | Status | Report |
|--------|------|--------|--------|
| **Sprint 0** | Establish testing infrastructure & baseline metrics | ✅ Complete | [SPRINT_0_REPORT.md](./SPRINT_0_REPORT.md) |
| **Sprint 1** | Implement TurboQuant KV cache compression | 🔄 In Progress | — |
| **Sprint 2+** | Optimize for multi-GPU, edge devices | 📋 Planned | — |

---

## Quick Start

### Run Baseline Locally

```bash
./scripts/run-on-mac.sh
```

Downloads latest release, validates binaries, runs smoke test, then optionally runs baseline stress test with 7B model.

**Requirements:**
- macOS or Linux
- 10GB free RAM (for 7B baseline)
- `gh` CLI (GitHub Actions integration) - logged in
- ollama installed and logged in
- hf cli installed and logged in

### Verify Release

```bash
./scripts/verify-release.sh
```

Checks latest GitHub release for binaries and test artifacts.

### Run Smoke Test

```bash
./scripts/smoke-test.sh --build-dir bin
```

Quick 10-token inference test (18MB model, ~2 seconds).

---

## Architecture

**Architect:** Mickey Perlstein

**Core Components:**
- **Inference Engine:** llama.cpp (C++ with Metal GPU support)
- **KV Compression:** Learned quantization during attention
- **Testing:** Real-world datasets (WikiText-103), multi-part reasoning tasks
- **CI/CD:** GitHub Actions (automated builds, benchmarks, releases)

---

## Baseline Metrics (Sprint 0)

**Memory-Bound Stress Test**
- Model: Mistral-7B-Instruct-v0.1 (Q4_0)
- Context: 1810 words from WikiText-103-v1
- Task: Multi-part summarization + analysis (1000 tokens)
- Configuration: 8K context window
- **Result: 7.05 tokens/sec**
- Status: Memory-bound (KV cache is bottleneck)

Why this baseline matters: Large context + large output = massive KV cache. Compression gains will be dramatic.

---

## Build from Source

```bash
# Configure (Metal GPU on macOS)
cmake -B build -S llama.cpp \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j$(nproc)

# Test
ctest --test-dir build --output-on-failure
```

---

## License

Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License)

---

## References

1. [llama.cpp](https://github.com/ggerganov/llama.cpp) — Efficient inference framework
2. [Ollama](https://ollama.ai) — Model serving
3. [Hugging Face](https://huggingface.co) — Models & datasets
4. [KV Cache Quantization Research](https://arxiv.org/abs/2404.09668) — Google Research
5. [WikiText-103 Dataset](https://huggingface.co/datasets/wikitext) — Real-world text corpus
