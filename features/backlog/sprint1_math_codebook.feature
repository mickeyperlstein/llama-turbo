Feature: Sprint 1 — TurboQuant Math & Codebook in C++

  Implement the core TurboQuant algorithms in C++ with Python cross-validation
  against the turboquant_plus reference implementation.

  Background:
    Given the TurboQuant paper algorithms are understood
    And the turboquant_plus Python reference implementation is available
    And a C++ build environment with CMake is configured

  Scenario: Random orthogonal rotation matrix generation
    Given a head_dim of 128
    When a random orthogonal rotation matrix R is generated with a fixed seed
    Then R has shape [128, 128]
    And R * R^T equals the identity matrix within float16 precision
    And the same seed always produces the same matrix

  Scenario: Max-Lloyd codebook generation for Beta distribution
    Given the rotation induces a Beta distribution on each coordinate
    When the Max-Lloyd algorithm solves the 1D optimal scalar quantizer
    Then a codebook of 2^(b-1) centroids is produced for b=4 (turbo4)
    And a codebook of 2^(b-1) centroids is produced for b=3 (turbo3)
    And codebook values are baked as static const float arrays in codebook.cpp
    And centroid values match turboquant_plus reference within 1e-4 tolerance

  Scenario: Johnson-Lindenstrauss sign residual
    Given a residual vector of dimension 128
    And a random Gaussian projection matrix S
    When the JL residual is computed as sign(S * residual)
    Then the output is a 1-bit vector of length equal to the projection dimension
    And the inner product estimate is unbiased
    And the result matches turboquant_plus reference output for identical inputs

  Scenario: Full TurboQuant encode roundtrip
    Given a KV tensor of shape [1, 40, 512, 128] in float16
    When turbo4 encode is applied
    Then output size is <= original_size / 3.8
    And roundtrip MSE after decode is < 0.01
    And roundtrip MSE matches turboquant_plus Python reference within 5%

  Scenario: Python cross-validation via pybind11
    Given the C++ implementation is compiled as a Python module via pybind11
    When the same synthetic KV tensor is encoded by both C++ and turboquant_plus
    Then MSE between the two outputs is < 1e-4
    And the test is runnable in GitHub Actions on ubuntu-latest without GPU
