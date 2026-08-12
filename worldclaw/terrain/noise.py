"""Seeded coherent noise bands.

``N_rk`` in the height-field sum.  Frequencies are given in cycles across the
world extent, so the same spec produces the same landform at any raster
resolution -- only the sampling gets finer.

Two implementation choices worth stating, because both are visible in the
output:

* **Gradient (Perlin) noise, not value noise.**  Value noise on a lattice puts
  its extrema *on* the lattice points, and terracing or erosion downstream turns
  that into a visible rectangular maze.  Gradient noise puts zeros on the
  lattice instead, which reads as organic.
* **Each octave is rotated and offset.**  Even gradient noise leaks its axes
  when octaves stack in register; rotating by an irrational-ish angle per octave
  breaks the alignment for free.
"""

from __future__ import annotations

import numpy as np

from ..schemas import NoiseComponent

# Golden angle: successive octaves never come back into alignment.
_OCTAVE_ROTATION = np.deg2rad(137.507764)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Ken Perlin's C2-continuous fade, 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lattice(n: int, gen: np.random.Generator) -> np.ndarray:
    """(n, n, 2) unit gradient vectors, wrapping at the edges."""
    ang = gen.random((n, n), dtype=np.float32) * np.float32(2.0 * np.pi)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)


def _perlin_at(uy: np.ndarray, ux: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Evaluate 2D gradient noise at arbitrary lattice coordinates.

    Coordinates wrap modulo the lattice size, so callers may rotate or translate
    the sampling grid freely without running off the array.
    """
    n = grad.shape[0]
    iy = np.floor(uy).astype(np.int32)
    ix = np.floor(ux).astype(np.int32)
    ty = (uy - iy).astype(np.float32)
    tx = (ux - ix).astype(np.float32)
    iy0, ix0 = iy % n, ix % n
    iy1, ix1 = (iy0 + 1) % n, (ix0 + 1) % n

    def dot(gy, gx, oy, ox):
        g = grad[gy, gx]
        return g[..., 0] * ox + g[..., 1] * oy

    d00 = dot(iy0, ix0, ty, tx)
    d01 = dot(iy0, ix1, ty, tx - 1.0)
    d10 = dot(iy1, ix0, ty - 1.0, tx)
    d11 = dot(iy1, ix1, ty - 1.0, tx - 1.0)

    fy, fx = _smootherstep(ty), _smootherstep(tx)
    top = d00 + fx * (d01 - d00)
    bot = d10 + fx * (d11 - d10)
    # 2D Perlin peaks at sqrt(2)/2; rescale so the band is roughly [-1, 1].
    return ((top + fy * (bot - top)) * np.float32(np.sqrt(2.0))).astype(np.float32)


def gradient_noise(
    shape: tuple[int, int],
    frequency: float,
    gen: np.random.Generator,
    rotation: float = 0.0,
) -> np.ndarray:
    """One band of gradient noise with ``frequency`` cycles per world edge."""
    h, w = shape
    n = max(2, int(round(frequency)))
    grad = _lattice(n, gen)

    # Sample in [0, n) across both axes, then rotate about the centre.
    y = np.linspace(0.0, n, h, endpoint=False, dtype=np.float32)[:, None]
    x = np.linspace(0.0, n, w, endpoint=False, dtype=np.float32)[None, :]
    y = np.broadcast_to(y, shape)
    x = np.broadcast_to(x, shape)
    if rotation:
        c, s = np.float32(np.cos(rotation)), np.float32(np.sin(rotation))
        cy = cx = np.float32(n / 2.0)
        y, x = (
            cy + (y - cy) * c - (x - cx) * s,
            cx + (y - cy) * s + (x - cx) * c,
        )
    # Random per-band phase so two bands of equal frequency are not identical.
    off = gen.random(2, dtype=np.float32) * n
    return _perlin_at(y + off[0], x + off[1], grad)


def fbm(
    shape: tuple[int, int],
    frequency: float,
    gen: np.random.Generator,
    octaves: int = 1,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    kind: str = "perlin",
) -> np.ndarray:
    """Octave sum of gradient noise, normalised back to roughly [-1, 1].

    ``ridged`` inverts the absolute value (sharp crests, rounded valleys) and
    ``billow`` does the opposite (rounded hills, creased valleys); both are the
    standard remappings from the procedural-terrain literature.  The remap is
    applied per octave, not to the sum -- that is what makes ridged noise carve
    branching gullies instead of one big crease.
    """
    total = np.zeros(shape, dtype=np.float32)
    amp, freq, norm = 1.0, float(frequency), 0.0
    for o in range(octaves):
        band = gradient_noise(shape, freq, gen, rotation=o * _OCTAVE_ROTATION)
        if kind == "ridged":
            band = 1.0 - 2.0 * np.abs(band)
        elif kind == "billow":
            band = 2.0 * np.abs(band) - 1.0
        total += amp * band
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / max(norm, 1e-6)).astype(np.float32)


def noise_stack(
    shape: tuple[int, int], components: list[NoiseComponent], root_seed: int, region: str
) -> np.ndarray:
    """Evaluate ``sum_k w_rk * N_rk(x)`` for one region."""
    from ..rng import rng as make_rng

    out = np.zeros(shape, dtype=np.float32)
    for k, c in enumerate(components):
        gen = make_rng(root_seed, "noise", region, k, c.kind, c.frequency)
        out += c.weight * fbm(
            shape, c.frequency, gen, c.octaves, c.lacunarity, c.gain, kind=c.kind
        )
    return out
