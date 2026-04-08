# =============================================================================
# Dockerfile — llama-turbo Linux development and CI image
#
# Provides a reproducible build environment with all C++ and Python deps.
# Used by:
#   - CI workflows (ci.yml, release.yml) via Ubuntu container
#   - Local development: docker build -t llama-turbo . && docker run llama-turbo
#
# Metal is OFF — this image is for Linux CPU-only builds.
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Install C++ toolchain and Python bindings ---
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    git \
    python3 \
    python3-pip \
    python3-dev \
    pybind11-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    pytest

WORKDIR /workspace

# --- Copy repo and build (no Metal on Linux) ---
COPY . .

RUN cmake -B build \
    -DGGML_METAL=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

# --- Default: run tests ---
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
