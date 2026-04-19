# TurboQuant: KV Cache Compression Algorithm

**Source:** Google Research ICLR 2026, arXiv 2504.19874  
**Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)  
**Blog:** [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## Overview

TurboQuant is a **post-training, calibration-free, online KV cache quantization** method that:
- Compresses K and V heads independently at write time (when tokens are cached)
- Requires no training data, model fine-tuning, or per-layer calibration
- Achieves **4–6× memory reduction** (2.5–3.5 bits/value) with minimal quality loss
- Works via learned rotation (PolarQuant) + optional 1-bit residual correction (QJL)

In practice, **MSE-only mode (PolarQuant) is recommended** — the QJL residual correction adds variance that softmax amplifies, degrading attention quality. See community testing in [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969).

---

## Algorithm: TurboQuant-MSE (PolarQuant)

### Core Insight

Raw KV vectors have **outlier coordinates** (some dimensions much larger than others). A random orthogonal rotation spreads energy evenly. After rotation, each dimension approximately follows a **Beta distribution** that in high dimensions converges to **N(0,1)**.

Because all coordinates now have identical marginal distribution, use **one precomputed Lloyd-Max codebook** for every coordinate. No calibration data needed.

### Encode (Write Path)

```
Input:  float16 vector x of dimension d (one K or V head)
Output: block_tbq3_0 or block_tbq4_0

1. Compute L2 norm:      norm = ‖x‖₂
2. Normalize:            x_unit = x / norm
3. Zero-pad to power-of-2: n = next_pow2(d)

4. Apply HD³ rotation:
   For r = 0..2:
     - Apply random sign flips: x_unit[i] *= (–1)^bit(i)  using splitmix64(seed⊕r)
     - Apply in-place WHT: butterfly on unnormalized x_unit
   - Divide all by n^(3/2)   (normalize 3 rounds of WHT)

5. Scale up:    val[j] = x_unit[j] * √n    (maps N(0, 1/n) → N(0,1))

6. Quantize:    idx[j] = argmin_k |val[j] – centroid_k|
   - 3-bit: 8 centroids (∈ {–2.15, –1.34, –0.76, –0.25, 0.25, 0.76, 1.34, 2.15})
   - 4-bit: 16 centroids (evenly spaced from –2.73 to +2.73)

7. Pack:        block.qs = packed_indices[QK_K]
                block.d = FP16(norm)  [block 0 only; others set d=0]
```

### Decode (Read Path)

```
Input:  block array for a KV row
Output: reconstructed F16 vector of dimension d

1. Read norm from block[0].d

2. For each block b:
   - Unpack indices from b.qs
   - Look up centroids: buf[b*QK_K + j] = codebook[idx[j]]
   - Scale down:        buf[...] *= (1/√n)

3. Zero-fill padding:   buf[d..n–1] = 0

4. Apply inverse HD³ (reverse order):
   For r = 2..0:
     - Apply in-place WHT
     - Apply same sign flips as round r (self-inverse)
   - Divide all by n^(3/2)

5. Scale by norm:       y[i] = buf[i] * norm
```

### Rotation: HD³ (Subsampled Randomized Hadamard)

**Why not dense random matrix?** O(d²) space; high variance. HD³ is the solution:

- **Walsh-Hadamard Transform** — O(d log d) butterfly, exact orthogonal transform
- **Random sign flips** — break symmetry before each WHT round
- **3 rounds** — empirically optimal for convergence to Haar distribution

```c
void wht_inplace(float *x, int64_t n) {
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += len << 1) {
            for (int64_t j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

void apply_sign_flip(float *x, int64_t n, uint64_t seed) {
    uint64_t state = seed;
    for (int64_t i = 0; i < n; i += 64) {
        uint64_t bits = splitmix64_next(&state);
        for (int64_t j = i; j < i+64 && j < n; j++) {
            if (bits & 1) x[j] = -x[j];
            bits >>= 1;
        }
    }
}

void srht_forward(float *x, int64_t n, uint64_t base_seed) {
    for (int r = 0; r < 3; r++) {
        apply_sign_flip(x, n, derive_round_seed(base_seed, r));
        wht_inplace(x, n);
    }
    float norm_factor = powf((float)n, -1.5f);
    for (int64_t i = 0; i < n; i++) x[i] *= norm_factor;
}

void srht_inverse(float *x, int64_t n, uint64_t base_seed) {
    for (int r = 2; r >= 0; r--) {
        wht_inplace(x, n);
        apply_sign_flip(x, n, derive_round_seed(base_seed, r));
    }
    float norm_factor = powf((float)n, -1.5f);
    for (int64_t i = 0; i < n; i++) x[i] *= norm_factor;
}
```

**PRNG: splitmix64**
```c
static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
```

### Codebooks (Lloyd-Max, Hardcoded)

Precomputed once for N(0,1) distribution. No calibration data.

**3-bit (8 centroids):**
```
Centroids:  {–2.1520, –1.3440, –0.7560, –0.2451, 0.2451, 0.7560, 1.3440, 2.1520}
Boundaries: {–1.7480, –1.0500, –0.5006, 0.0000, 0.5006, 1.0500, 1.7480}
```

**4-bit (16 centroids):**
```
Centroids:  {–2.7326, –2.0690, –1.6180, –1.2562, –0.9424, –0.6568, –0.3881, –0.1284,
              0.1284,  0.3881,  0.6568,  0.9424,  1.2562,  1.6180,  2.0690,  2.7326}
Boundaries: {–2.4008, –1.8435, –1.4371, –1.0993, –0.7996, –0.5225, –0.2583, 0.0000,
              0.2583,  0.5225,  0.7996,  1.0993,  1.4371,  1.8435,  2.4008}
```

**Lookup:** simple boundary scan:
```c
uint8_t quantize_scalar(float val, const float *boundaries, int n_boundaries) {
    for (int i = 0; i < n_boundaries; i++) {
        if (val < boundaries[i]) return (uint8_t)i;
    }
    return (uint8_t)n_boundaries;
}
```

---

## Block Format (Official PR #21089)

### TBQ3_0 — 3.0625 bits/element (5.2× vs F16)

```c
typedef struct {
    uint8_t   qs[QK_K * 3 / 8];  // 96 bytes: 256 values × 3-bit, LSB-packed
    ggml_half d;                  // 2 bytes: FP16 L2 norm (block 0 only)
} block_tbq3_0;  // sizeof = 98 bytes
// Bitwidth: 98 × 8 / 256 = 3.0625 bits/value
```

### TBQ4_0 — 4.0625 bits/element (3.9× vs F16)

```c
typedef struct {
    uint8_t   qs[QK_K / 2];  // 128 bytes: 256 values × 4-bit as nibbles
    ggml_half d;              // 2 bytes: FP16 L2 norm (block 0 only)
} block_tbq4_0;  // sizeof = 130 bytes
// Bitwidth: 130 × 8 / 256 = 4.0625 bits/value
```

**Key:** norm stored once per row (in block[0].d), not per block, because HD³ rotation is per-row.

---

## Compression Ratios

| Format | Bits/value | Compression vs F16 |
|--------|-----------|---------------------|
| F16 | 16.0 | 1× |
| TBQ3_0 | 3.0625 | 5.2× |
| TBQ4_0 | 4.0625 | 3.9× |
| TQ3_0 + QJL | 3.5 | 4.6× |

The "6× compression" claim from the blog comes from **2.5-bit mixed mode**: 32 outlier channels at 3-bit + 96 channels at 2-bit = (32×3 + 96×2) / 128 = 2.5 bits/value. This has ~1.2% relative PPL degradation on LongBench.

---

## Theoretical Bounds (Validation)

For unit-norm vectors:

| Bitwidth | MSE Bound |
|----------|-----------|
| 3-bit | D_mse ≤ 0.034 per dimension |
| 4-bit | D_mse ≤ 0.009 per dimension |

Your C++ implementation should hit these numbers within 1%.

---

## QJL: 1-Bit Residual Correction (Optional)

This is **Stage 2** of TurboQuant. Includes 1 extra bit per dimension to correct systematic bias in inner product estimation.

**In practice:** Usually skipped. Community benchmarks on llama.cpp show QJL hurts attention quality because softmax amplifies the added variance. MSE-only is the validated approach.

---

## Integration with Flash Attention

Two paths exist:

**Non-fused (dequantize-then-dot):** Dequantize K/V to F16, run normal attention. Simpler, used in CPU path.

**Fused (dot in compressed space):** Pre-rotate Q by the same HD³. Compute Q_rot · codebook_lookup(K_codes) directly. This requires:
1. Rotating Q at each attention step by same HD³ transform
2. Implementing vec_dot kernels that dot a rotated Q against codebook-indexed K

Formula preserved under orthogonal transforms:
```
⟨Q, K⟩ = ⟨Π·Q, Π·K_unit⟩ · ‖K‖
```

---

## Key Properties

| Property | Value |
|----------|-------|
| Post-training? | Yes, applied purely at inference |
| Calibration data? | None — ever |
| Rotation seed | Fixed per (layer_idx, head_idx); deterministic |
| Rotation storage | None (implicit via seed; O(d log d) time) |
| Data-oblivious? | Yes — codebooks don't depend on model weights or data |
| Supports F16→Q8 chains? | Yes, and Q8→Q4 cascade as well |

---

## Reference Implementations

| Repo | Notes |
|------|-------|
| `mudler/llama.cpp @ dee102d` | Cleanest C reference; ground truth |
| `ggml-org/llama.cpp #21089` | Official upstream PR; CPU-first, TBQ3_0/TBQ4_0 |
| `unixsysdev/llama-turboquant` | TQ3_0 with QJL residual, 32-value blocks |
| `scos-lab/turboquant` | Python/NumPy reference; 2,500 LOC, 49 tests |
| `lingengyuan/qjl-mlx` | MLX/Apple Silicon with QJL |

---

## Community Discussion

[llama.cpp #20969 — TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Extensive ablation studies
- Comparison: WHT vs random rotation (WHT wins 59.65× better PPL)
- K/V asymmetry recommendations
- QJL quality impact analysis
