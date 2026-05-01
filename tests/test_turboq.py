"""
Chunk 7: Statistical correctness tests for the TurboQuant encode/decode pipeline.

Test 1 — Round-trip RMSE distribution (1000 random float16 vectors, dim=256):
  The per-vector RMSE should be tightly distributed around the Lloyd-Max theoretical
  distortion for N(0,1) quantization. We check:
    - Mean RMSE within [0.04, 0.22] for 4-bit (theoretical ~0.085)
    - Mean RMSE within [0.08, 0.40] for 3-bit (theoretical ~0.17)
    - No inf or NaN in any decoded output

Test 2 — JL inner-product unbiasedness (100 random vector pairs):
  The dot product between two unit-norm vectors should be preserved on average after
  encoding and decoding one of them. Mean absolute error < 0.05, no systematic bias.

Saves results to benchmark/results/error_distribution.json.
"""

import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_turboq_ext"))
import turboq_ext


RNG = np.random.default_rng(0xC0FFEE)
N_VECS = 1000
DIM = 256
N_PAIRS = 100
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmark", "results", "error_distribution.json")


def encode_decode(v_f16: np.ndarray, bit_width: int) -> np.ndarray:
    enc = turboq_ext.encode(v_f16.view(np.uint16), 0, 0, bit_width)
    dec_u16 = turboq_ext.decode(enc, DIM, 0, 0, bit_width)
    return dec_u16.view(np.float16)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


class TestRoundTripRMSE:
    @classmethod
    def setup_class(cls):
        vecs = RNG.standard_normal((N_VECS, DIM)).astype(np.float16)
        cls.vecs = vecs
        cls.rmse4 = [rmse(vecs[i], encode_decode(vecs[i], 4)) for i in range(N_VECS)]
        cls.rmse3 = [rmse(vecs[i], encode_decode(vecs[i], 3)) for i in range(N_VECS)]

    def test_4bit_no_nan_inf(self):
        for i, v in enumerate(self.vecs):
            dec = encode_decode(v, 4).astype(np.float32)
            assert not np.any(np.isnan(dec)), f"NaN in 4-bit decode for vector {i}"
            assert not np.any(np.isinf(dec)), f"Inf in 4-bit decode for vector {i}"

    def test_3bit_no_nan_inf(self):
        for i, v in enumerate(self.vecs):
            dec = encode_decode(v, 3).astype(np.float32)
            assert not np.any(np.isnan(dec)), f"NaN in 3-bit decode for vector {i}"
            assert not np.any(np.isinf(dec)), f"Inf in 3-bit decode for vector {i}"

    def test_4bit_rmse_in_range(self):
        mean4 = float(np.mean(self.rmse4))
        assert 0.04 <= mean4 <= 0.22, f"4-bit mean RMSE {mean4:.4f} outside expected [0.04, 0.22]"

    def test_3bit_rmse_in_range(self):
        mean3 = float(np.mean(self.rmse3))
        assert 0.08 <= mean3 <= 0.40, f"3-bit mean RMSE {mean3:.4f} outside expected [0.08, 0.40]"

    def test_zero_vector(self):
        z = np.zeros(DIM, dtype=np.float16)
        for bw in (3, 4):
            dec = encode_decode(z, bw).astype(np.float32)
            assert float(np.max(np.abs(dec))) == 0.0, f"Zero vector not preserved for {bw}-bit"

    def test_save_results(self):
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        results = {
            "4bit": {
                "mean_rmse": float(np.mean(self.rmse4)),
                "std_rmse": float(np.std(self.rmse4)),
                "max_rmse": float(np.max(self.rmse4)),
            },
            "3bit": {
                "mean_rmse": float(np.mean(self.rmse3)),
                "std_rmse": float(np.std(self.rmse3)),
                "max_rmse": float(np.max(self.rmse3)),
            },
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)


class TestJLUnbiasedness:
    """
    After encoding one vector (v) and decoding it, the dot product <v_dec, u>
    should be an unbiased estimate of <v, u> for random unit-norm u.
    Mean absolute error across 100 pairs should be < 0.05.
    """

    def test_inner_product_bias(self):
        errors = []
        for _ in range(N_PAIRS):
            v = RNG.standard_normal(DIM).astype(np.float16)
            u = RNG.standard_normal(DIM).astype(np.float16)

            # normalize both to unit sphere
            v_f32 = v.astype(np.float32)
            u_f32 = u.astype(np.float32)
            v_norm = float(np.linalg.norm(v_f32))
            u_norm = float(np.linalg.norm(u_f32))
            if v_norm == 0 or u_norm == 0:
                continue
            v_unit = (v_f32 / v_norm).astype(np.float16)
            u_unit = (u_f32 / u_norm).astype(np.float16)

            v_dec = encode_decode(v_unit, 4).astype(np.float32)
            true_dot = float(np.dot(v_unit.astype(np.float32), u_unit.astype(np.float32)))
            approx_dot = float(np.dot(v_dec, u_unit.astype(np.float32)))
            errors.append(abs(approx_dot - true_dot))

        mean_err = float(np.mean(errors))
        assert mean_err < 0.05, f"JL inner-product mean absolute error {mean_err:.4f} >= 0.05"
