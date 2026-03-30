Feature: Benchmark Robustness and Anti-Gaming

  Ensure benchmark results reflect real-world performance.
  Strengthens the Sprint 4 kill switch against agent optimization tricks.

  Background:
    Given Qwen2.5-Coder-32B on Apple Silicon Max Pro 32GB
    And Metal backend confirmed active
    And baseline f16 results are available for comparison

  Scenario: Throughput must not regress for memory gains
    Given turbo4 achieves memory reduction >= 30% vs baseline
    Then throughput must be >= baseline throughput - 3%
    And trading throughput for memory below this floor is a NO-GO

  Scenario: Tail latency constraint
    Given 20 benchmark runs minimum for statistical validity
    When P95 latency is computed across all runs
    Then P95 latency must not exceed baseline P95 by more than 10%
    And raw latency distribution is saved to benchmark/results/latency_distribution.json

  Scenario: Long context stability at 4096 tokens
    Given a 4096 token generation run with turbo4
    Then no NaN or degenerate outputs occur
    And output coherence is maintained throughout
    And results are saved to benchmark/results/long_context_4096.json

  Scenario: Long context stability at 8192 tokens
    Given an 8192 token generation run with turbo4
    Then no NaN or degenerate outputs occur
    And attention distribution does not degrade vs 4096 token run
    And results are saved to benchmark/results/long_context_8192.json

  Scenario: Regression detection with tight threshold
    Given previous benchmark results exist in benchmark/results/
    When new benchmark results are produced on the self-hosted runner
    Then regression > 3% in any metric fails CI
    And the failing metric and delta are reported in the CI summary
    And results are never overwritten without a passing CI run
