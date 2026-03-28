Feature: Sprint 5 — Ollama Integration & Upstream PR

  Wire TurboQuantKVCache through Ollama's kvcache.Cache Go interface
  and prepare upstream PRs for both llama.cpp and Ollama.

  Background:
    Given Sprint 4 returned a GO decision
    And benchmark data is available in benchmark/results/summary.md
    And Ollama's kvcache.Cache Go interface is understood
    And llama.cpp discussion has been opened with benchmark data

  Scenario: Ollama kvcache.Cache interface implemented
    Given Ollama's existing kvcache.Cache interface
    When a TurboCache struct is implemented in kvcache/turbo.go
    Then it satisfies the kvcache.Cache interface
    And it delegates compress/decompress to the llama.cpp C layer via CGo
    And go build succeeds with no errors

  Scenario: turbo4 selectable via Ollama model parameters
    Given an Ollama Modelfile
    When PARAMETER kv_cache_type turbo4 is set
    Then Ollama passes --cache-type-k turbo4 to the llama.cpp backend
    And ollama run with the model uses TurboQuantKVCache internally

  Scenario: Ollama integration tests pass
    Given a small test model (7B Q4_K_M) for speed
    When ollama run is executed with turbo4 cache type
    Then the model produces coherent output
    And no crashes or NaN outputs occur over 20 sequential prompts
    And kvcache tests in the Ollama test suite pass

  Scenario: llama.cpp upstream PR prepared
    Given the implementation is complete and benchmarked
    When the llama.cpp PR is drafted
    Then it targets ggml-org/llama.cpp main branch
    And it adds TurboQuantKVCache as a new llama_memory_i implementation
    And it includes benchmark results table in the PR description
    And it follows llama.cpp contribution guidelines
    And a discussion thread is opened first with benchmark data before the PR

  Scenario: Ollama upstream PR prepared
    Given the llama.cpp PR is open or has maintainer signal
    When the Ollama PR is drafted
    Then it targets ollama/ollama main branch
    And it wires the new cache type through Ollama's parameter system
    And it references the llama.cpp PR for context
    And CI passes on the Ollama PR

  Scenario: Self-hosted benchmark runner registered
    Given the Max Pro 32GB machine
    When registered as a GitHub Actions self-hosted runner for mickeyperlstein/llama-turbo
    Then benchmark jobs trigger automatically on merge to main
    And results are committed back as artifacts
    And a regression threshold check fails the job if tok/s drops > 5% vs previous run
