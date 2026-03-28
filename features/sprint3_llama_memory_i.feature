Feature: Sprint 3 — llama_memory_i Implementation

  Implement TurboQuantKVCache as a concrete llama_memory_i class,
  selectable via --cache-type-k turbo4 in the llama.cpp CLI.

  Background:
    Given Sprint 2 artifacts are available: ggml ops, Metal shader, pybind11 bindings
    And llama_memory_i abstract interface exists in llama.cpp (PR #12181)
    And existing concrete implementations are reference: llama_kv_cache, llama_memory_recurrent

  Scenario: TurboQuantKVCache class implements llama_memory_i
    Given the llama_memory_i interface contract
    When TurboQuantKVCache is implemented in turbo_kv_cache.cpp
    Then all pure virtual methods of llama_memory_i are implemented
    And the class compiles with no warnings

  Scenario: Cache stores compressed codes not full float16 tensors
    Given a KV update call with key tensor shape [1, 40, 1, 128]
    When TurboQuantKVCache.update() is called
    Then the stored representation uses <= original_bytes / 3.8
    And get_memory_bytes() reflects the compressed size

  Scenario: Decompress on read returns correct full KV history
    Given a 512-step decode sequence stored in TurboQuantKVCache
    When the full KV history is retrieved for attention computation
    Then the decompressed tensors have shape [1, 40, 512, 128]
    And values match a float16 reference cache within roundtrip MSE < 0.01

  Scenario: turbo4 selectable via --cache-type-k flag
    Given llama.cpp CLI argument parsing
    When --cache-type-k turbo4 is passed
    Then TurboQuantKVCache is instantiated instead of the default llama_kv_cache
    And --cache-type-k turbo3 selects 3.5-bit compression mode
    And --cache-type-k f16 still selects the original uncompressed cache

  Scenario: First and last layers excluded from compression by default
    Given a 64-layer model
    When TurboQuantKVCache is initialized with default config
    Then layers 0 and 63 use uncompressed float16 KV storage
    And layers 1-62 use turbo4 compressed storage
    And skip_layers is configurable via --cache-turbo-skip-layers

  Scenario: GQA models handled correctly
    Given Qwen2.5-Coder-32B uses Grouped Query Attention with num_kv_heads != num_q_heads
    When TurboQuantKVCache is initialized with the model config
    Then num_kv_heads is used for cache head dimension, not num_q_heads
    And no shape mismatch errors occur during inference

  Scenario: Rotation matrix seed is consistent across encode and decode
    Given a fixed seed per (layer_idx, head_idx)
    When encode is called at step 1 and decode is called at step 100
    Then the same rotation matrix is used in both calls
    And no silent corruption occurs from seed mismatch
