# Contributing to llama-turbo

## Overview

This project implements TurboQuant KV cache compression as a `llama_memory_i` implementation for llama.cpp. All work is defined via Gherkin feature files and executed by agents or engineers sprint by sprint.

---

## Repository Structure

```
features/
  backlog/   — defined, not yet started. Agents may read, not act.
  current/   — active sprint. One sprint active at a time.
  closed/    — completed and verified. Read-only. Report attached.

benchmark/
  results/   — all benchmark output JSON files
  datasets/  — dataset references and pinned versions

docs/        — architecture and design notes
.github/
  workflows/ — CI (ubuntu, macos) and benchmark (self-hosted) pipelines
```

---

## Feature File Lifecycle

```
backlog/ → current/ → closed/
```

- **Sprint start**: move the sprint's feature files from `backlog/` to `current/`
- **Sprint end**: move feature files from `current/` to `closed/` with an attached report
- **Agents working in `current/`** own those files for the duration of the sprint
- **Agents must not modify files in `backlog/` or `closed/`**

---

## Closing a Feature

A feature may only move to `closed/` when ALL of the following are true:

1. All scenarios in the feature file pass
2. A report file exists alongside the feature: `closed/sprint_N_feature_name_report.md`
3. The report contains:
   - Full debug logs from all runs
   - Hardware spec (chip, memory, OS version, Metal version)
   - Exact software versions (llama.cpp commit hash, Python version, mlx version if used)
   - All benchmark results as raw data, not summaries
   - Dataset references (see below)
   - Reproduction instructions: exact commands to rerun and get identical results

---

## Reproducibility Standard

All results must be independently reproducible by any peer with equivalent hardware.

### Dataset Pinning

Every test that uses a dataset must reference it exactly:

| Dataset | Source | How to pin |
|---|---|---|
| HumanEval | `openai/human-eval` | Pin commit hash in report |
| WikiText-2 | HuggingFace `wikitext`, `wikitext-2-raw-v1` | Pin dataset revision |
| Synthetic tensors | Generated in-repo | Commit seed + generation script to `benchmark/datasets/` |

### What "reproducible" means

A peer should be able to:
1. Clone this repo at the commit referenced in the report
2. Follow the reproduction instructions exactly
3. Obtain results within the stated variance tolerance

If they cannot — the result does not count and the feature cannot be closed.

---

## Sprint Execution Order

Sprints are strictly sequential. A sprint agent must not begin until the previous sprint's features are all in `closed/`.

```
Sprint 0 — DevOps & CI foundation (gates everything)
Sprint 1 — TurboQuant math & codebook in C++
Sprint 2 — ggml ops & Metal shader
Sprint 3 — llama_memory_i implementation
Sprint 4 — Benchmarking & go/no-go (kill switch)
Sprint 5 — Upstream PR to ggml-org/llama.cpp
```

Sprint 4 is the kill switch. If go/no-go criteria are not met, the project stops at Sprint 4. A full benchmark report is still written and committed — a negative result with good data is a valid outcome.

---

## Artifact Contracts Between Sprints

Each sprint consumes the previous sprint's closed artifacts.

| Sprint | Requires from previous |
|---|---|
| Sprint 1 | Sprint 0: CI green, build working, self-hosted runner active |
| Sprint 2 | Sprint 1: `codebook.cpp`, `rotation.cpp`, `jl_residual.cpp`, pybind11 bindings, tests passing |
| Sprint 3 | Sprint 2: ggml ops, Metal shader, roundtrip tests passing |
| Sprint 4 | Sprint 3: `TurboQuantKVCache` integrated, `--cache-type-k turbo4` working in llama-cli |
| Sprint 5 | Sprint 4: GO decision, benchmark report, `RECOMMENDATION.md` |

---

## IoC Contract Rule

The KV cache abstraction is the foundation of this project. It must be real, not a mirage.

Every `llama_memory_i` implementation must:
- Override `update()` and log a write to `logs/kv_access.log` confirming float dependency
- Never expose raw float16 tensor references outside the cache class
- Pass `sprint0_kv_abstraction_integrity.feature` before any Sprint 3 work begins

---

## CI Rules

- `features/backlog/` and `features/closed/` are read-only in CI — any modification fails the build
- Threshold values in `tests/thresholds.json` are immutable without an accompanying benchmark data PR
- CPU fallback to NumPy on any GPU test is a hard CI failure
- Regression > 3% on any benchmark metric fails the self-hosted runner job

---

## Implementation Repo

All C++ implementation work happens in the fork:
**`mickeyperlstein/llama.cpp`**

The upstream PR target is `ggml-org/llama.cpp`. Open a discussion thread with benchmark data before opening a PR.
