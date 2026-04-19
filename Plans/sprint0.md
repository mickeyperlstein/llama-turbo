# Plan: --turbo-quant Plugin Flag

## Context

llama-turbo wraps llama.cpp and implements TurboQuant KV cache compression.
The KV cache is already fully quantization-aware — `cache_type_k` and `cache_type_v`
can be any supported ggml_type (Q8_0, Q4_0, etc.) and GGML handles the conversion
at eval time via `ggml_set_rows`. The quantization pipeline is already there;
we just need a single convenience flag that activates it.

## Approach

**Sprint 0: Flag infrastructure (this task)**
Add `--turbo-quant` as a no-op flag that does nothing. Just registers in the arg system so we can:
1. Verify the flag parser accepts it
2. Run inference with `--turbo-quant` and confirm no crashes
3. Prove the flag infrastructure works

**Sprint 1: Implement actual compression**
Once the flag works, swap the no-op with real TurboQuant (PolarQuant + QJL).

## Files to Modify

### 1. `llama.cpp/common/common.h` — line ~540
Add `bool turbo_quant = false;` to `common_params` near other KV-related bools:
```cpp
bool no_kv_offload     = false;
bool turbo_quant       = false; // --turbo-quant: KV cache compression (Sprint 1 implementation)
bool kv_unified        = false;
```

### 2. `llama.cpp/common/arg.cpp` — after line 2020 (after --cache-type-v block)
Add the flag as a no-op (Phase 1 only):
```cpp
add_opt(common_arg(
    {"--turbo-quant"},
    {"--no-turbo-quant"},
    "enable TurboQuant KV cache compression (default: disabled)",
    [](common_params & params, bool value) {
        params.turbo_quant = value;
        // Sprint 0: no-op. Sprint 1 will implement PolarQuant + QJL here.
    }
).set_env("LLAMA_ARG_TURBO_QUANT"));
```

## Why This Approach

**Sprint 0: No-op flag validates infrastructure**
- Separates "does the flag system work?" from "does the compression work?"
- Guarantees clean baseline: any Sprint 1 issues are compression-specific, not flag-related
- Fast to implement and test

**Sprint 1: Implement TurboQuant — 6x compression**
- PolarQuant: random rotation to redistribute variance evenly
- QJL transform: 1-bit error correction (Johnson-Lindenstrauss)
- Target: 3.5-4 bits/value (vs 16-bit baseline)
- Builds on proven flag infrastructure

## Key Constraint (Sprint 1)

`type_v` quantization requires flash attention. llama-context.cpp enforces this at
lines 349-353 and 2967-2970. When we implement TurboQuant in Sprint 1, we'll auto-enable
`flash_attn = true` in the flag handler so users don't have to pass both flags.

## Verification (Sprint 0)

1. Build: `cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)`
2. Flag exists: `bin/llama-cli --help | grep turbo-quant`
3. No crashes: `./scripts/smoke-test.sh --build-dir bin`
4. Baseline unchanged: `./scripts/run-on-mac.sh baseline --turbo-quant` outputs same tokens/sec as without flag (no-op confirmed)

## Files NOT Modified

- `src/llama-kv-cache.h` — no change needed
- `src/llama-kv-cache.cpp` — no change needed
- `src/llama-graph.cpp` — no change needed
- `src/llama-context.cpp` — no change needed (FA guard already exists)
