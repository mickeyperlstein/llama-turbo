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

  Scenario: Docker build for local C++ development
    Given Docker is installed on the developer machine
    When docker build -t llama-turbo . is run from the repo root
    Then the image builds successfully
    And the image contains: cmake, clang, python3, pybind11, ctest
    And docker run llama-turbo ctest --test-dir build runs all unit tests
    And no C++ toolchain installation is required on the host machine
    And the Dockerfile lives at repo root and is maintained alongside CI

  Scenario: Docker image used in ubuntu CI job
    Given .github/workflows/ci.yml ubuntu-latest job
    When the CI job runs
    Then it builds and runs tests inside the Docker image
    And the same Dockerfile used locally is used in CI
    And local and CI results are identical for the same commit

  Scenario: GitHub Release assets on merge to main
    Given a merge to main succeeds with green CI
    When the release workflow triggers
    Then a GitHub Release is created tagged with the commit SHA
    And the following artifacts are attached:
      | llama-turbo-linux-x86_64.tar.gz   | Linux build binaries + .so |
      | llama-turbo-macos-arm64.tar.gz    | macOS Metal build binaries |
      | benchmark-results.json            | Latest benchmark output    |
      | test-report.xml                   | ctest XML results          |
    And release notes list passing and failing scenarios from the sprint
    And artifacts are downloadable without authentication

  Scenario: Sprint 1 agent readiness check
    Given all previous Sprint 0 scenarios pass
    When the Sprint 1 agent starts
    Then it can clone mickeyperlstein/llama.cpp and build it
    And it can run existing llama.cpp unit tests successfully
    And it has access to turboquant_plus Python reference via requirements.txt
    And CI is green on both ubuntu-latest and macos-latest
