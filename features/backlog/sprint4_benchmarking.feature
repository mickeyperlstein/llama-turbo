Feature: Sprint 4 — Benchmarking & Go/No-Go

  Benchmark TurboQuant KV cache against baseline on Apple Silicon Max Pro 32GB.
  This sprint is the kill switch — results determine whether Sprint 5 proceeds.

  Background:
    Given Sprint 3 artifacts are available: TurboQuantKVCache integrated in llama.cpp
    Given Qwen2.5-Coder-32B GGUF weights are downloaded at Q4_K_M
    And the Apple Silicon Max Pro 32GB is the exclusive benchmark machine
    And no background processes are running during benchmarks
    And Metal backend is confirmed active

  Scenario: Baseline throughput measurement
    Given llama.cpp running with --cache-type-k f16 (uncompressed)
    When 5 prompts are run at output lengths 256, 1024, and 2048 tokens
    Then tokens/sec is recorded for each length
    And time-to-first-token is recorded
    And peak Metal memory is recorded via mx.metal.get_active_memory()
    And results are saved to benchmark/results/baseline_throughput.json

  Scenario: turbo4 throughput measurement
    Given llama.cpp running with --cache-type-k turbo4
    When the identical 5 prompts and lengths as baseline are run
    Then tokens/sec, TTFT, and peak memory are recorded
    And results are saved to benchmark/results/turbo4_throughput.json

  Scenario: turbo3 throughput measurement
    Given llama.cpp running with --cache-type-k turbo3
    When the identical benchmark suite is run
    Then results are saved to benchmark/results/turbo3_throughput.json

  Scenario: HumanEval coding quality measurement
    Given a 50-problem subset of the HumanEval benchmark
    When pass@1 is evaluated for baseline, turbo4, and turbo3
    Then results are saved to benchmark/results/{format}_humaneval.json
    And quality delta vs baseline is computed

  Scenario: Go criteria evaluation
    Given all benchmark results are collected
    When the go/no-go criteria are applied
    Then the result is GO if:
      | throughput improvement >= 5% at 2048 token output |
      | OR peak memory reduction >= 30%                   |
      | AND pass@1 delta vs baseline <= 3 points          |
      | AND zero NaN or degenerate outputs across all runs |
    And the result is NO-GO otherwise
    And RECOMMENDATION.md is written with decision, data, and rationale

  @experimental
  Scenario: Attention without full decode — compressed-space inner product
    Given TurboQuant's unbiased inner-product estimation property from the paper
    And the JL projection preserves inner products in expectation
    When attention scores Q·K^T are computed directly on compressed K codes
    Then no full float16 K tensor is materialized during the attention operation
    And attention scores are estimated as Q · (R^T · codebook_lookup(codes)) in one fused op
    And V is still decompressed for the weighted sum (partial decode only)
    And output quality is measured via HumanEval pass@1 against full-decode baseline
    And latency is measured — hypothesis: memory bandwidth reduction compounds with compute reduction
    And results are saved to benchmark/results/experimental_nodecode.json
    And the scenario is marked LEAP if pass@1 delta < 3 points AND tok/s > turbo4 by >= 10%
    And the scenario is marked VALIDATED_PATH if it fails — confirming full decode is necessary

  Scenario: Benchmark results published as CI artifact
    Given the self-hosted runner on Max Pro completes the benchmark suite
    When results are pushed to the benchmark/results/ directory
    Then a summary table is generated in benchmark/results/summary.md
    And the CI job fails if any format shows regression > 5% vs its previous run
