Feature: Statistical Properties Validation

  Validate that quantization preserves expected statistical structure.
  Prevents "looks right but isn't" math bugs in the core algorithms.

  Background:
    Given the TurboQuant C++ implementation from Sprint 1
    And pybind11 bindings expose encode/decode to Python test harness

  Scenario: Quantization error distribution matches expected variance
    Given TurboQuant encoding applied to 1000 random float16 vectors of dimension 128
    When reconstruction error is measured per vector
    Then the error distribution variance is within 10% of the theoretical Beta distribution variance
    And the error mean is within 1% of zero (unbiased)
    And results are saved to benchmark/results/error_distribution.json

  Scenario: JL projection unbiasedness
    Given 100 random vector pairs (x, y) of dimension 128
    When inner product is estimated via JL residual projection for each pair
    Then the mean dot product error across all 100 samples is approximately zero
    And the absolute mean error is < 0.01
    And no systematic bias direction is detectable across the sample set
    And results are saved to benchmark/results/jl_unbiasedness.json
