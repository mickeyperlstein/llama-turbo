Feature: KV Cache Strategy Integrity

  Ensure the KV cache abstraction fully controls representation and access.
  Validates the IoC contract is real, not a leaky wrapper.

  Scenario: KV implementation does not leak raw tensor assumptions
    Given TurboQuantKVCache is active
    When attention requests KV tensors
    Then no direct float16 tensor references are exposed outside the cache class
    And all KV access goes through abstraction methods only

  Scenario: Memory layout varies by strategy
    Given multiple KV strategies are initialized with identical model config:
      | f16    |
      | turbo4 |
    When storing identical KV updates
    Then underlying memory layout differs between strategies
    And get_memory_bytes() reflects actual storage for each strategy
    And the difference in get_memory_bytes() matches the expected compression ratio

  @experimental
  Scenario: No implicit full decode during compressed-space attention
    Given TurboQuantKVCache is active
    And the experimental compressed-space attention path is enabled via --cache-turbo-nodecode
    When attention is executed
    Then no full KV tensor reconstruction occurs unless explicitly requested
    And decode operations are instrumented and counted per attention step
    And implicit full decode triggers a test failure in this mode
