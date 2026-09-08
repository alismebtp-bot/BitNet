#!/usr/bin/env python3
"""Verify I2_S pack/unpack layout (128 weights / 32-byte interleaved groups).

Pure-stdlib so CI unit tests need no third-party packages.
"""

from __future__ import annotations

import random
import unittest


def pack_i2_s(q: list[int]) -> list[int]:
    """Pack ternary codes {0,1,2} with official BitNet layout."""
    assert len(q) % 128 == 0
    n_blocks = len(q) // 128
    packed: list[int] = []
    for b in range(n_blocks):
        base = b * 128
        for gp in range(32):
            byte = (
                (q[base + gp] << 6)
                | (q[base + 32 + gp] << 4)
                | (q[base + 64 + gp] << 2)
                | q[base + 96 + gp]
            )
            packed.append(byte & 0xFF)
    return packed


def unpack_i2_s_interleaved(packed: list[int], n: int) -> list[int]:
    """Correct scalar unpack used by the portable mad.cpp fallback."""
    out = [0] * n
    for k in range(n):
        B = k // 128
        c = (k % 128) // 32
        gp = (k % 128) % 32
        b = packed[B * 32 + gp]
        out[k] = (b >> (6 - 2 * c)) & 3
    return out


def unpack_i2_s_contiguous_wrong(packed: list[int], n: int) -> list[int]:
    """Incorrect contiguous-nibble scheme (4 consecutive elems per byte)."""
    out = [0] * n
    for i in range(n):
        byte_idx = i // 4
        bit_pos = 6 - 2 * (i % 4)
        out[i] = (packed[byte_idx] >> bit_pos) & 3
    return out


class I2SLayoutTests(unittest.TestCase):
    def test_roundtrip_interleaved(self):
        rng = random.Random(0)
        q = [rng.randrange(0, 3) for _ in range(128 * 5)]
        packed = pack_i2_s(q)
        self.assertEqual(len(packed), 32 * 5)
        recovered = unpack_i2_s_interleaved(packed, len(q))
        self.assertEqual(recovered, q)

    def test_contiguous_differs(self):
        rng = random.Random(1)
        q = [rng.randrange(0, 3) for _ in range(128)]
        packed = pack_i2_s(q)
        wrong = unpack_i2_s_contiguous_wrong(packed, len(q))
        self.assertNotEqual(wrong, q)

    def test_dot_matches_reference(self):
        rng = random.Random(2)
        q = [rng.randrange(0, 3) for _ in range(256)]
        acts = [rng.randrange(-8, 9) for _ in range(256)]
        packed = pack_i2_s(q)
        decoded = unpack_i2_s_interleaved(packed, len(q))
        self.assertEqual(
            sum(d * a for d, a in zip(decoded, acts)),
            sum(qq * a for qq, a in zip(q, acts)),
        )


if __name__ == "__main__":
    unittest.main()
