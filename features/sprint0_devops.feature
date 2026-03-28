Feature: Sprint 0 — DevOps & CI Foundation

  Establish the full development environment, CI pipeline, and repo structure
  before any implementation begins. Sprint 1 agents must find everything ready.

  Background:
    Given mickeyperlstein/llama-turbo is the coordination repo
    And mickeyperlstein/llama.cpp is the implementation fork
    And the Apple Silicon Max Pro 32GB is the benchmark machine

  Scenario: llama.cpp fork builds cleanly
    Given mickeyperlstein/llama.cpp is forked from ggml-org/llama.cpp
    When the project is built with Metal support enabled
      | cmake -B build -DLLAMA_METAL=ON |
      | cmake --build build --config Release |
    Then the build succeeds with no errors
    And llama-cli binary is produced
    And llama-cli --version runs successfully

  Scenario: GitHub Actions CI on ubuntu-latest
    Given .github/workflows/ci.yml exists in mickeyperlstein/llama-turbo
    When a push is made to any branch
    Then the workflow triggers on ubuntu-latest
    And llama.cpp fork is checked out as a submodule or dependency
    And cmake build completes successfully
    And placeholder unit test suite runs and passes
    And build artifacts are cached between runs

  Scenario: GitHub Actions CI on macos-latest
    Given .github/workflows/ci.yml includes a macos-latest job
    When a push is made to any branch
    Then the workflow triggers on macos-latest
    And cmake build with -DLLAMA_METAL=ON completes successfully
    And Metal shader compilation produces no errors
    And placeholder unit test suite runs and passes

  Scenario: Self-hosted runner registered on Max Pro
    Given the Apple Silicon Max Pro 32GB machine
    When registered as a GitHub Actions self-hosted runner
      | label: self-hosted  |
      | label: macos        |
      | label: apple-silicon|
      | label: benchmark    |
    Then the runner appears as active in mickeyperlstein/llama-turbo settings
    And a smoke job triggers on the runner and completes successfully
    And the runner is isolated to benchmark-tagged jobs only

  Scenario: Benchmark CI job skeleton
    Given .github/workflows/benchmark.yml exists
    When triggered manually via workflow_dispatch or on merge to main
    Then the job runs on the self-hosted apple-silicon runner
    And it builds llama.cpp with Metal support
    And it runs a 10-token smoke generation with a small test model
    And it outputs a benchmark/results/smoke.json artifact
    And the job fails if the smoke generation produces no output

  Scenario: Repo structure scaffolded in llama-turbo
    Given mickeyperlstein/llama-turbo
    When Sprint 0 is complete
    Then the following directories exist:
      | features/     | Gherkin sprint definitions        |
      | benchmark/    | Benchmark harness and results     |
      | docs/         | Architecture and design notes     |
      | .github/workflows/ | CI and benchmark workflows   |
    And a CONTRIBUTING.md defines the sprint agent workflow

  Scenario: llama.cpp fork sync workflow
    Given ggml-org/llama.cpp continues active development
    When .github/workflows/sync-upstream.yml is configured
    Then a weekly scheduled job opens a PR to merge upstream main into mickeyperlstein/llama.cpp
    And merge conflicts are flagged for manual resolution
    And CI must pass on the sync PR before merge

  Scenario: Sprint 1 agent readiness check
    Given all previous Sprint 0 scenarios pass
    When the Sprint 1 agent starts
    Then it can clone mickeyperlstein/llama.cpp and build it
    And it can run existing llama.cpp unit tests successfully
    And it has access to turboquant_plus Python reference via requirements.txt
    And CI is green on both ubuntu-latest and macos-latest
