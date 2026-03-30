@experimental
Feature: Compressed Attention Feasibility

  Validate whether attention inner products can be computed directly
  in compressed space without full KV decode. Only meaningful once
  Sprint 4 benchmark data exists — do not attempt before Sprint 4.

  Background:
    Given Sprint 3 TurboQuantKVCache is complete and benchmarked
    And Sprint 4 baseline throughput data is available
    And TurboQuant's unbiased inner-product estimation property is understood

  Scenario: Attention scores estimated in compressed space
    Given TurboQuantKVCache is active with --cache-turbo-nodecode flag
    When Q·K^T attention scores are computed
    Then no full float16 K tensor is materialized
    And scores are estimated via Q · (R^T · codebook_lookup(codes))
    And V is still decompressed for the weighted sum (partial decode only)

  Scenario: Quality gate for compressed-space attention
    Given compressed-space attention is active
    When HumanEval pass@1 is measured
    Then quality delta vs full-decode turbo4 is < 3 points
    And results are saved to benchmark/results/experimental_nodecode.json

  Scenario: Leap or validated path outcome
    Given all experimental measurements are complete
    Then result is LEAP if pass@1 delta < 3 points AND tok/s > turbo4 by >= 10%
    And result is VALIDATED_PATH if it fails — full decode confirmed necessary
    And either outcome is a valid and documented result
