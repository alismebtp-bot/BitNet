#!/usr/bin/env python3
"""Verify I2_S pack/unpack layout (128 weights / 32-byte interleaved groups)."""

from __future__ import annotations

import unittest

import numpy as np


def pack_i2_s(q: np.ndarray) -> np.ndarray:
    """Pack ternary codes {0,1,2} with official BitNet layout."""
    assert q.ndim == 1 and q.size % 128 == 0
    n_blocks = q.size // 128
    q = q.reshape(n_blocks, 4, 32)
    packed = (q[:, 0, :] << 6) | (q[:, 1, :] << 4) | (q[:, 2, :] << 2) | q[:, 3, :]
    return packed.astype(np.uint8).reshape(-1)


def unpack_i2_s_interleaved(packed: np.ndarray, n: int) -> np.ndarray:
    """Correct scalar unpack used by the portable mad.cpp fallback."""
    out = np.empty(n, dtype=np.int32)
    for k in range(n):
        B = k // 128
        c = (k % 128) // 32
        gp = (k % 128) % 32
        b = int(packed[B * 32 + gp])
        out[k] = (b >> (6 - 2 * c)) & 3
    return out


def unpack_i2_s_contiguous_wrong(packed: np.ndarray, n: int) -> np.ndarray:
    """Incorrect contiguous-nibble scheme (4 consecutive elems per byte)."""
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        byte_idx = i // 4
        bit_pos = 6 - 2 * (i % 4)
        out[i] = (int(packed[byte_idx]) >> bit_pos) & 3
    return out


class I2SLayoutTests(unittest.TestCase):
    def test_roundtrip_interleaved(self):
        rng = np.random.default_rng(0)
        q = rng.integers(0, 3, size=128 * 5, dtype=np.int32)
        packed = pack_i2_s(q)
        self.assertEqual(packed.size, 32 * 5)
        recovered = unpack_i2_s_interleaved(packed, q.size)
        np.testing.assert_array_equal(recovered, q)

    def test_contiguous_differs(self):
        rng = np.random.default_rng(1)
        q = rng.integers(0, 3, size=128, dtype=np.int32)
        packed = pack_i2_s(q)
        wrong = unpack_i2_s_contiguous_wrong(packed, q.size)
        # Contiguous decode must not match the official interleaved packing.
        self.assertFalse(np.array_equal(wrong, q))

    def test_dot_matches_reference(self):
        rng = np.random.default_rng(2)
        q = rng.integers(0, 3, size=256, dtype=np.int32)
        acts = rng.integers(-8, 9, size=256, dtype=np.int32)
        packed = pack_i2_s(q)
        decoded = unpack_i2_s_interleaved(packed, q.size)
        self.assertEqual(int(np.dot(decoded, acts)), int(np.dot(q, acts)))


if __name__ == "__main__":
    unittest.main()
