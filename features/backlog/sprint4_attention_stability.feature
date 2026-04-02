Feature: Attention Stability Under Long Context

  Validate that TurboQuant does not introduce compounding error
  in attention behavior over long token sequences.

  Background:
    Given Sprint 3 artifacts are available: TurboQuantKVCache integrated in llama.cpp
    And deterministic inference mode is enabled with a fixed seed
    And Qwen2.5-Coder-32B is the test model

  Scenario: Attention distribution stability vs baseline
    Given a fixed seed and deterministic inference mode
    And a prompt of length 512 tokens
    When generating 2048 tokens with:
      | cache-type-k | f16    |
      | cache-type-k | turbo4 |
    Then attention logits are captured at layers [10, 20, 30]
    And KL divergence between turbo4 and f16 attention distributions is < 0.05
    And top-5 attention indices overlap >= 80% per head
    And results are saved to benchmark/results/attention_stability.json

  Scenario: Logit drift over generation steps
    Given identical prompt and seed
    When generating 2048 tokens with turbo4 and f16
    Then per-token logit difference (L2) is recorded at every step
    And the drift does not grow superlinearly across steps
    And no monotonic drift pattern emerges
    And drift curve is saved to benchmark/results/logit_drift.json

  Scenario: Output token divergence bounded
    Given identical prompt and seed
    When generating 1024 tokens with turbo4 and f16
    Then token match rate between turbo4 and f16 is >= 70%
    And perplexity difference is within 10%
    And results are saved to benchmark/results/token_divergence.json

  Scenario: Stability results feed go/no-go gate
    Given all attention stability scenarios have run
    When go/no-go evaluation in sprint4_benchmarking.feature is applied
    Then KL divergence > 0.05 at any sampled layer is a NO-GO signal
    And superlinear logit drift is a NO-GO signal regardless of throughput gains
