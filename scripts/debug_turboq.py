"""Debug script to trace turboq encode/decode pipeline and find inf/NaN source."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build_turboq_ext'))

import turboq_ext
import numpy as np

rng = np.random.default_rng(42)

# Simple unit vector — norm=1, normalized already
v = np.zeros(256, dtype=np.float16)
v[0] = 1.0

v_u16 = v.view(np.uint16)

# Encode 4-bit
enc4 = turboq_ext.encode(v_u16, 0, 0, 4)   # breakpoint here

# Peek at the encoded block: d (norm, fp16) + qs (128 bytes)
import struct
norm_bits = struct.unpack_from('<H', enc4, 128)[0]  # d is last 2 bytes (offset 128)
norm_f32 = struct.unpack_from('<e', enc4, 128)[0]
print(f"encoded norm (fp16 bits=0x{norm_bits:04x}, f32={norm_f32:.6f})")
print(f"first 8 qs bytes: {list(enc4[:8])}")

# Decode
dec4_u16 = turboq_ext.decode(enc4, 256, 0, 0, 4)
dec4 = dec4_u16.view(np.float16)

print(f"decoded[0:4] = {dec4[:4]}")
print(f"decoded max abs = {float(np.max(np.abs(dec4.astype(np.float32)))):.4f}")

has_inf = np.any(np.isinf(dec4.astype(np.float32)))
has_nan = np.any(np.isnan(dec4.astype(np.float32)))
print(f"has_inf={has_inf}  has_nan={has_nan}")

if not has_inf and not has_nan:
    rmse = float(np.sqrt(np.mean((v.astype(np.float32) - dec4.astype(np.float32))**2)))
    print(f"RMSE = {rmse:.4f}")
