Feature: Sprint 2 — ggml Ops & Metal Shader

  Wrap TurboQuant C++ math as ggml operations with a Metal shader
  for Apple Silicon GPU execution.

  Background:
    Given Sprint 1 artifacts are available: codebook.cpp, rotation.cpp, jl_residual.cpp
    And pybind11 bindings from Sprint 1 are passing
    And ggml source is available as a dependency
    And an Apple Silicon Mac with Metal support is available for shader testing

  Scenario: ggml op registration for turbo encode
    Given the TurboQuant encode logic from Sprint 1
    When wrapped as a custom ggml op GGML_OP_TURBO_ENCODE
    Then the op accepts a float16 ggml tensor as input
    And returns a quantized ggml tensor with compressed codes
    And the op is registered in ggml.h and ggml.c

  Scenario: ggml op registration for turbo decode
    Given a compressed ggml tensor from GGML_OP_TURBO_ENCODE
    When GGML_OP_TURBO_DECODE is applied
    Then the output float16 tensor matches the original within roundtrip MSE < 0.01
    And the op is registered in ggml.h and ggml.c

  Scenario: Metal shader for encode on Apple Silicon
    Given ggml-metal.metal exists in the llama.cpp source
    When a Metal kernel turbo_encode_kernel is added
    Then the kernel executes on Apple Silicon GPU via Metal
    And output matches CPU reference implementation within float16 precision
    And Metal backend is confirmed active (not CPU fallback) during execution

  Scenario: Metal shader for decode on Apple Silicon
    Given a compressed tensor produced by turbo_encode_kernel
    When turbo_decode_kernel executes on Metal GPU
    Then the decoded tensor matches CPU decode within float16 precision
    And latency per decode step for shape [1, 40, 1, 128] is measured and recorded

  Scenario: pybind11 bindings expose ggml ops to Python
    Given the ggml ops are implemented
    When compiled with pybind11
    Then Python can call turbo_encode(np.ndarray) -> np.ndarray
    And Python can call turbo_decode(np.ndarray) -> np.ndarray
    And roundtrip test passes on ubuntu-latest in GitHub Actions without GPU

  Scenario: CI build validation
    Given the CMakeLists.txt is updated with new source files
    When the project is built on ubuntu-latest GitHub Actions runner
    Then the build succeeds with no errors
    And all unit tests pass
    And Metal shader compilation is validated on macos-latest runner
