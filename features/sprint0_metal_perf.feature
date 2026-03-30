Feature: Metal Kernel Performance Validation

  Ensure TurboQuant Metal kernels provide meaningful performance.
  Correct but useless GPU work is a CI failure.

  Background:
    Given turbo_encode_kernel and turbo_decode_kernel are implemented
    And Metal backend is confirmed active (not CPU fallback)
    And the Apple Silicon Max Pro 32GB is the test machine

  Scenario: Encode kernel throughput baseline
    Given the target KV tensor shape [1, 40, 512, 128] in float16
    When turbo_encode_kernel executes on Metal GPU
    Then total execution time is less than a full memcpy of the uncompressed tensor
    And execution time is recorded to benchmark/results/metal_perf.json

  Scenario: Bitpacking efficiency
    Given a compressed KV tensor in turbo4 format
    When stored in memory
    Then effective bytes per element is <= 0.5 bytes (4 bits)
    And memory alignment overhead wastes no more than 10% above theoretical minimum

  Scenario: GPU vs CPU at realistic decode shape
    Given both CPU reference and Metal implementations of encode/decode
    When running identical encode/decode at shape [1, 40, 512, 128]
    Then Metal GPU latency is less than CPU latency
    And the comparison is recorded at seq >= 256 — the realistic decode threshold
    And GPU advantage is logged per sequence length: 256, 512, 1024
