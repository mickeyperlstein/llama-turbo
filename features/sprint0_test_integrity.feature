Feature: Test Integrity Enforcement

  Prevent agents from weakening tests or bypassing constraints.
  All scenarios enforced from Sprint 0 — CI fails on any violation.

  Scenario: Assertion thresholds are immutable
    Given baseline test thresholds are defined in tests/thresholds.json
    When a PR modifies any test file
    Then any increase in tolerance thresholds causes CI failure
    And threshold changes require accompanying benchmark data in the PR description
    And a separate reviewer approval is required for any threshold relaxation

  Scenario: Test input diversity enforced
    When running unit tests
    Then at least 10 inputs are used per test case
    And at least 3 different seeds are used across the input set
    And the test must pass on all inputs, not just the majority

  Scenario: No-op implementation detection
    Given TurboQuant encode is implemented
    When encoding a tensor
    Then output entropy differs from input by a minimum threshold
    And output is not bitwise identical to input
    And compression ratio is >= 2.0x or the test fails

  Scenario: CPU fallback detection
    Given Metal backend is required for GPU tests
    When any GPU test runs
    Then execution must confirm Metal device is active via metal device query
    And CPU fallback triggers immediate test failure
    And the Metal device name is logged in test output for audit

  @experimental
  Scenario: No implicit full decode during compressed-space attention
    Given TurboQuantKVCache is active
    And the experimental compressed-space attention path is enabled
    When attention is executed
    Then no full KV tensor reconstruction occurs unless explicitly requested
    And decode operations are instrumented and logged per step
    And any implicit full decode triggers a test warning flagged in CI output
