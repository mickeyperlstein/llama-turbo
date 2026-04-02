FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    git \
    python3 \
    python3-pip \
    python3-dev \
    pybind11-dev \
    libpybind11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    pytest

WORKDIR /workspace

# Copy repo and build
COPY . .

RUN cmake -B build \
    -DLLAMA_METAL=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
