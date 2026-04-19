# Sprint 1 Plan: TurboQuant KV Cache Compression

**→ See [Sprint 1 HLD](../docs/sprint1-hld.md) for architecture diagrams and data flow**

## Context

Sprint 0 locked baseline at 7.05–8.06 tokens/sec on Mistral-7B (8K context, memory-bound).
Target is 2–3× improvement via TurboQuant/PolarQuant online KV cache quantization.

A prior agent already added `block_tbq3_0` / `block_tbq4_0` structs and `QK_TBQ3_0` / `QK_TBQ4_0`
defines to `ggml.h` and incremented `GGML_TYPE_COUNT`. The encode/decode math, type trait
registration, integration hook, pybind11 bindings, and statistical tests remain.

---

## Algorithm Reference

**PolarQuant encode (per KV row, e.g. head_dim=128):**
1. Compute L2 norm; normalize to unit sphere
2. Zero-pad to next power of 2
3. Apply HD3: 3 rounds of (splitmix64 sign flip → in-place WHT → divide by n^(3/2))
4. Scale by sqrt(n) → N(0,1) distribution
5. Quantize each coordinate via argmin against Lloyd-Max centroids; pack indices
6. Store norm in `block[0].d` (fp16)

**Decode:** reverse — unpack, centroid lookup, scale down, inverse HD3, scale by norm.

**Codebooks (hardcoded):**
- 3-bit (8 centroids): `{-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520}`
- 4-bit (16 centroids): `{-2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326}`

**Block formats** (from `ggml-common.h`, already scaffolded):
- `block_tbq3_0`: 98 bytes = `qs[96]` + `d`(fp16) — 3.0625 bits/elem, 5.2× vs F16
- `block_tbq4_0`: 130 bytes = `qs[128]` + `d`(fp16) — 4.0625 bits/elem, 3.9× vs F16
- Block size QK = 256 values; norm stored once per row in `block[0].d`

**PRNG seed:** `splitmix64`, seeded per `(layer_idx, head_idx)` — deterministic, no calibration.

---

## Files to Create / Modify

**Dedicated TurboQuant module** (isolated, easy to rebase):

| File | Action | Purpose |
|------|--------|---------|
| `turboq/turboq.h` | Create | Public API: `turboq_encode_f16`, `turboq_decode_f16`, `turboq_register_type_traits` |
| `turboq/turboq.c` | Create | Core encode/decode logic + type trait registration function |
| `turboq/codebook.h` | Create | Static const centroid arrays for 3-bit and 4-bit |
| `turboq/hd3.h` | Create | splitmix64 + WHT + HD3 rotation interface |
| `turboq/hd3.c` | Create | HD3 implementation |
| `turboq/CMakeLists.txt` | Create | Build turboq as static/shared library, pybind11 target |
| `turboq/bindings.cpp` | Create | pybind11 module `turboq_ext` exposing encode/decode |
| `tests/test_turboq.py` | Create | Statistical correctness tests (pytest) |

**Minimal shared code changes:**

| File | Action | Purpose |
|------|--------|---------|
| `llama.cpp/src/llama-context.cpp` | Modify | Call `turboq_register_type_traits()` at startup; pass `GGML_TYPE_TBQ4_0` as `type_k` (guarded by flag) |
| `CMakeLists.txt` | Modify | Add `turboq` as subdirectory (conditional `-DLLAMA_TURBOQ=ON`) |

---

## Implementation Chunks

### Chunk 1 — Codebook (codebook.h / codebook.cpp)
**Inputs:** hardcoded centroid values from docs/turboquant-algorithm.md  
**Outputs:** `extern const float TURBOQ_CENTROIDS_3BIT[8]`, `extern const float TURBOQ_CENTROIDS_4BIT[16]`  
**Test:** C unit test: values match within 1e-4 of spec; 3-bit has 8 entries, 4-bit has 16

### Chunk 2 — HD3 Rotation (hd3.h / hd3.c)
**Inputs:** head_dim, layer_idx, head_idx  
**Outputs:** in-place rotation of float* buf, inverse rotation  
**Algorithm:** splitmix64 seeded by `(layer_idx << 16 | head_idx)`, 3 rounds of sign-flip + WHT + normalize  
**Test:** `R * R^T ≈ I` within float16 precision; same seed → same result; matches turboquant_plus reference

### Chunk 3 — Quantize/Dequantize Functions (turboq/turboq.c)
**Location:** isolated in `turboq/turboq.c`, no ggml-quants.c changes  
**New functions:**
```c
void turboq_encode_f16(const ggml_half * x, void * y, int64_t n, int bit_width, 
                       uint32_t layer_idx, uint32_t head_idx);
void turboq_decode_f16(const void * x, ggml_half * y, int64_t n, int bit_width,
                       uint32_t layer_idx, uint32_t head_idx);
```
**Test:** roundtrip RMSE < 0.01; output size ≤ original/3.8; same as existing quant tests

### Chunk 4 — Type Traits Registration (turboq/turboq.c)
**Pattern:** `turboq_register_type_traits()` function that registers TBQ3_0/TBQ4_0 at runtime  
**Called from:** `llama-context.cpp` startup (guarded by `#ifdef LLAMA_TURBOQ`)  
**Registration:** custom `to_float` / `from_float_ref` callbacks → `turboq_decode_f16` / `turboq_encode_f16`  
**Constraint:** `n_embd_head_k` (128 for Mistral-7B) must divide 256 — satisfied (128×2=256)  
**Test:** `ggml_type_size(GGML_TYPE_TBQ4_0)` returns expected size; type is recognized as quantized

### Chunk 5 — KV Cache Integration (llama-context.cpp)
**Changes:**
  1. Call `turboq_register_type_traits()` early in context init (guarded by `#ifdef LLAMA_TURBOQ`)
  2. Set `params.type_k = GGML_TYPE_TBQ4_0` when `LLAMA_TURBOQ=ON` compile flag  
**Constraint:** quantized V requires Flash Attention (`-fa` flag); start with K-only compression  
**Test:** `./scripts/run-on-mac.sh` — confirm tokens/sec > 7.05; output quality unchanged (smoke test passes)

### Chunk 6 — pybind11 Bindings (turboq/bindings.cpp)
**Expose:** `encode(np.ndarray f16, layer_idx, head_idx, bit_width=4) -> bytes`, `decode(bytes, shape, layer_idx, head_idx, bit_width=4) -> np.ndarray`  
**Build:** pybind11 + CMake in `turboq/CMakeLists.txt`, CPU-only, runnable on ubuntu-latest in CI  
**Test:** same synthetic KV tensor [1,40,512,128] encoded by C++ and turboquant_plus — MSE < 1e-4

### Chunk 7 — Statistical Correctness Tests (tests/test_turboq.py)
**Test 1:** 1000 random float16 vectors (dim=128) → encode → decode → measure error variance within 10% of Beta distribution theory, mean within 1% of zero → save `benchmark/results/error_distribution.json`  
**Test 2:** 100 random vector pairs → JL residual inner product estimates → mean absolute error < 0.01, no systematic bias → save `benchmark/results/jl_unbiasedness.json`  
**CI:** add pytest step to `.github/workflows/ci.yml`

---

## Build Sequence

1. **Chunks 1–2** (codebook + HD3): independent, can be done in parallel
2. **Chunk 3** (encode/decode): depends on 1+2
3. **Chunk 4** (type traits registration): depends on 3
4. **Chunk 5** (KV integration): depends on 4; minimal change to shared code
5. **Chunk 6** (pybind11): depends on 3; isolated to `turboq/bindings.cpp`
6. **Chunk 7** (statistical tests): depends on 6

**Rebase-friendly:** all TurboQuant code lives in `turboq/` directory. Only integration point is `llama-context.cpp` calling `turboq_register_type_traits()` and passing `type_k=GGML_TYPE_TBQ4_0`.

---

## Verification

```bash
# Build
cmake -B build -S llama.cpp -DGGML_METAL=ON -DLLAMA_TURBOQ=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Unit tests
ctest --test-dir build --output-on-failure -R turbo

# Integration benchmark (must beat 7.05 tokens/sec)
./scripts/run-on-mac.sh

# Statistical correctness (Python, CPU-only)
pip install turboq_ext
pytest tests/test_turboq.py -v

# Cross-validate against turboquant_plus
python tests/validate_vs_reference.py
```

**Success criteria (from SPRINT_0_REPORT.md):**
- tokens/sec > 14 (2× baseline) — stretch goal 21 (3×)
- Smoke test passes (output quality unchanged)
- Statistical tests pass: variance within 10%, mean within 1% of zero
- Cross-validation MSE < 1e-4 vs turboquant_plus
