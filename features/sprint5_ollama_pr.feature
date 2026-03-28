Feature: Sprint 5 — Upstream PR to ggml-org/llama.cpp

  Prepare and submit the upstream PR from mickeyperlstein/llama.cpp
  to ggml-org/llama.cpp. Ollama requires no changes — it inherits
  the new --cache-type-k turbo4 option automatically via vendored llama.cpp.

  Background:
    Given Sprint 4 returned a GO decision
    And benchmark data is available in benchmark/results/summary.md
    And implementation is complete in mickeyperlstein/llama.cpp fork
    And a llama.cpp discussion thread has been opened with benchmark data

  Scenario: Ollama compatibility confirmed with no Go changes
    Given Ollama vendors llama.cpp and passes --cache-type-k flags through
    When turbo4 is merged into ggml-org/llama.cpp
    Then Ollama users get TurboQuant automatically on next llama.cpp vendor update
    And no Go code changes are required in ollama/ollama

  Scenario: llama.cpp upstream PR prepared
    Given the implementation is complete and benchmarked
    When the llama.cpp PR is drafted
    Then it targets ggml-org/llama.cpp main branch
    And it adds TurboQuantKVCache as a new llama_memory_i implementation
    And it includes benchmark results table in the PR description
    And it follows llama.cpp contribution guidelines
    And a discussion thread is opened first with benchmark data before the PR

  Scenario: Self-hosted benchmark runner registered
    Given the Max Pro 32GB machine
    When registered as a GitHub Actions self-hosted runner for mickeyperlstein/llama-turbo
    Then benchmark jobs trigger automatically on merge to main
    And results are committed back as artifacts
    And a regression threshold check fails the job if tok/s drops > 5% vs previous run
