"""Deterministic seed derivation.

Every stochastic step in the pipeline draws its generator from the run seed plus
a path of string/int labels (region name, operator index, ...).  Two runs with
the same seed and the same spec therefore produce bit-identical output, and
adding a region does not reshuffle the noise of its neighbours.
"""

from __future__ import annotations

import hashlib


def derive_seed(root_seed: int, *path: object) -> int:
    """Derive a stable 64-bit seed from a root seed and a label path."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(root_seed)).encode())
    for part in path:
        h.update(b"\x1f")
        h.update(str(part).encode())
    return int.from_bytes(h.digest(), "big")


def rng(root_seed: int, *path: object):
    """Return a numpy Generator seeded from ``root_seed`` and ``path``."""
    import numpy as np

    return np.random.default_rng(derive_seed(root_seed, *path))
