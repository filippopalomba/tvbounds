"""R-compatible random-number stream for the bootstrap.

Reproduces, bit for bit, the draws that R (>= 3.6.0, with its default
``RNGkind("Mersenne-Twister", "Inversion", "Rejection")``) produces after
``set.seed(seed)``, so that a seeded bootstrap in this package resamples
exactly the rows the R package resamples:

* ``set.seed(seed)`` scrambles the integer seed with the linear congruential
  step ``seed <- 69069 * seed + 1`` fifty times, then fills the 625-word
  Mersenne-Twister state with further steps; the first word (the position
  ``mti``) is reset to 624 by R's ``FixupSeeds``.
* ``unif_rand()`` returns ``fixup(genrand_int32() * 2.3283064365386963e-10)``.
* ``sample.int(n, size, replace = TRUE)`` draws ``R_unif_index(n) + 1`` for
  each element: rejection sampling from ``bits = ceil(log2(n))`` random bits
  assembled sixteen at a time from ``floor(unif_rand() * 65536)``.

Only the pieces the package needs are implemented (uniforms and sampling
with replacement); the generator is an instance, never global state, so a
seeded call has no side effect on NumPy's or Python's random modules.
"""
from __future__ import annotations

import math

import numpy as np

_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1), as in R's RNG.c
_MT_N = 624


class RRNG:
    """R's Mersenne-Twister stream after ``set.seed(seed)``."""

    def __init__(self, seed):
        seed = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            seed = (69069 * seed + 1) & 0xFFFFFFFF
        key = np.empty(_MT_N + 1, dtype=np.uint32)
        for j in range(_MT_N + 1):
            seed = (69069 * seed + 1) & 0xFFFFFFFF
            key[j] = seed
        # key[0] is R's dummy[0] = mti slot (reset to 624); mt = key[1:]
        bg = np.random.MT19937()
        bg.state = {"bit_generator": "MT19937",
                    "state": {"key": key[1:].copy(), "pos": _MT_N}}
        self._bg = bg

    # -- primitives ---------------------------------------------------------
    def unif_rand(self) -> float:
        """R's ``unif_rand()`` for Mersenne-Twister (in the open unit interval)."""
        v = float(self._bg.random_raw()) * 2.3283064365386963e-10
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def _rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        """R's ``R_unif_index(dn)`` under ``sample.kind = "Rejection"``."""
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    # -- user-level draws ---------------------------------------------------
    def runif(self, k: int) -> np.ndarray:
        """R's ``runif(k)``."""
        return np.array([self.unif_rand() for _ in range(int(k))],
                        dtype=np.float64)

    def sample_int_replace(self, n: int, size: int) -> np.ndarray:
        """R's ``sample.int(n, size, replace = TRUE)`` (1-based integers)."""
        n = int(n)
        size = int(size)
        out = np.empty(size, dtype=np.int64)
        dn = float(n)
        for i in range(size):
            out[i] = int(self.unif_index(dn)) + 1
        return out
