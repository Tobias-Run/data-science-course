"""Geomorphological operators ``G_rj``.

The paper names peak / dune / terrace / erosion but does not formulate them, so
these are our own formulations -- expect visual divergence from the published
figures.  They follow standard procedural-terrain practice (the paper points at
Patel 2015 for the same reason).

Contract for every operator:

* input  -- the regional field so far, ``h_r + sum_k w_rk N_rk(x)``, in
  normalised height units, plus the region's hard mask for spatial placement;
* output -- a *delta* in the same units, roughly in [-1, 1], to be scaled by
  ``alpha_rj`` and summed by the height field.

Returning a delta is what lets purely modifying operators (terrace, erosion)
live in the same additive sum as generative ones (peak, dune): ``alpha`` then
reads as "how much of this modification", and ``alpha = 0`` is always a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..rng import rng as make_rng
from ..schemas import (
    DuneOperator,
    ErosionOperator,
    PeakOperator,
    TerraceOperator,
)
from .noise import fbm


@dataclass
class OperatorContext:
    """Everything an operator may look at."""

    base: np.ndarray  # h_r + noise, normalised units
    mask: np.ndarray  # hard region mask (bool)
    cell_size_m: float  # world metres per pixel
    height_scale_m: float  # metres per normalised height unit
    world_size_m: float
    root_seed: int
    region: str
    index: int  # operator index within the region, for seed derivation

    def rng(self, *extra: object) -> np.random.Generator:
        return make_rng(self.root_seed, "op", self.region, self.index, *extra)

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# peak -- isolated summits, optionally merged into a ridge line
# ---------------------------------------------------------------------------


def op_peak(spec: PeakOperator, ctx: OperatorContext) -> np.ndarray:
    h, w = ctx.shape
    if not ctx.mask.any():
        return np.zeros(ctx.shape, dtype=np.float32)

    # Keep summits away from the region border: sample only where the distance
    # to the outside exceeds a fraction of the region's inradius.  A peak
    # straddling a boundary would be cut in half by the weight normalisation.
    dist = ndimage.distance_transform_edt(ctx.mask).astype(np.float32)
    threshold = spec.inset * float(dist.max())
    candidates = np.argwhere(dist >= max(threshold, 1.0))
    if len(candidates) == 0:
        candidates = np.argwhere(ctx.mask)

    gen = ctx.rng("centres")
    picks = gen.choice(len(candidates), size=min(spec.count, len(candidates)), replace=False)

    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    out = np.zeros(ctx.shape, dtype=np.float32)
    base_radius_px = spec.radius * max(h, w)

    for n, p in enumerate(picks):
        cy, cx = candidates[p]
        r_px = base_radius_px * float(1.0 + spec.radius_jitter * (gen.random() * 2 - 1))
        r_px = max(r_px, 2.0)
        amp = float(1.0 + spec.amplitude_jitter * (gen.random() * 2 - 1))

        t = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r_px
        np.clip(t, 0.0, 1.0, out=t)
        # cos profile: value 1 and zero gradient at the summit, C1 at the foot.
        # Clipped because cos(pi/2) evaluates a hair below zero in float32, and a
        # fractional sharpness on a negative base is NaN.
        bump = np.clip(np.cos(0.5 * np.pi * t), 0.0, 1.0) ** spec.sharpness
        out = np.maximum(out, amp * bump)  # merge, don't stack

    if spec.ridge_amount > 0:
        ridged = fbm(ctx.shape, spec.ridge_frequency, ctx.rng("ridge"), octaves=4, kind="ridged")
        out *= 1.0 - spec.ridge_amount * (0.5 - 0.5 * ridged)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# dune -- anisotropic ridges with a windward/lee asymmetry
# ---------------------------------------------------------------------------


def op_dune(spec: DuneOperator, ctx: OperatorContext) -> np.ndarray:
    h, w = ctx.shape
    yy = np.arange(h, dtype=np.float32)[:, None] / max(h, 1)
    xx = np.arange(w, dtype=np.float32)[None, :] / max(w, 1)

    theta = np.deg2rad(spec.direction_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)  # position across the crest lines
    phase = 2.0 * np.pi * u / spec.wavelength

    if spec.meander > 0:
        # Crest lines drift laterally instead of running dead straight.
        warp = fbm(ctx.shape, spec.meander_frequency, ctx.rng("meander"), octaves=3)
        phase = phase + 2.0 * np.pi * spec.meander * warp

    # sin(phi + a sin phi) skews the wave: gentle windward flank, steep slip
    # face -- the defining asymmetry of a migrating dune.
    profile = np.sin(phase + spec.asymmetry * np.sin(phase))
    profile = 0.5 * (profile + 1.0)  # dunes add relief, they do not dig
    return (profile ** spec.crest_sharpness).astype(np.float32)


# ---------------------------------------------------------------------------
# terrace -- stepped benches cut into the existing profile
# ---------------------------------------------------------------------------


def _step_blend(t: np.ndarray, sharpness: float) -> np.ndarray:
    """Map t in [0,1] to a soft step; sharpness 0 = identity, ->1 = hard edge."""
    p = 1.0 / max(1e-3, 1.0 - min(sharpness, 0.98))
    a = np.power(np.clip(t, 0.0, 1.0), p)
    b = np.power(np.clip(1.0 - t, 0.0, 1.0), p)
    return a / np.maximum(a + b, 1e-9)


def op_terrace(spec: TerraceOperator, ctx: OperatorContext) -> np.ndarray:
    u = ctx.base * spec.steps
    if spec.warp > 0:
        u = u + spec.warp * spec.steps * fbm(
            ctx.shape, spec.warp_frequency, ctx.rng("warp"), octaves=3
        )
    floor = np.floor(u)
    stepped = (floor + _step_blend(u - floor, spec.sharpness)) / spec.steps
    return (stepped - ctx.base).astype(np.float32)


# ---------------------------------------------------------------------------
# erosion -- thermal relaxation towards the angle of repose
# ---------------------------------------------------------------------------


def _shift(a: np.ndarray, dy: int, dx: int, fill: float | None = None) -> np.ndarray:
    """``out[i, j] = a[i - dy, j - dx]``, without wraparound.

    Out-of-range reads are clamped to the edge (``fill=None``) or replaced by
    ``fill``.  Erosion needs both: clamping when measuring height differences,
    so no artificial slope appears at the border and nothing flows off the map,
    and zero fill when redistributing, so no material arrives from outside.
    Using edge clamping for both is what makes a border row quietly manufacture
    material every iteration.
    """
    h, w = a.shape
    if fill is None:
        # Clamped-index gather: correct for any (dy, dx), including diagonals,
        # where a roll-then-patch approach fills corners from the wrong source.
        ry = np.clip(np.arange(h) - dy, 0, h - 1)
        cx = np.clip(np.arange(w) - dx, 0, w - 1)
        return a[np.ix_(ry, cx)]
    out = np.full_like(a, fill)
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


def op_erosion(spec: ErosionOperator, ctx: OperatorContext) -> np.ndarray:
    """Material above the repose angle slides to lower neighbours.

    Cheap, unconditionally stable for ``strength <= 1``, and it produces the
    talus fans and rounded scree slopes that pure noise never gives.  The talus
    threshold is a real gradient: it converts the repose angle through the cell
    size, so the same spec erodes consistently at any resolution.
    """
    h = ctx.base.astype(np.float32).copy()
    talus = np.float32(
        np.tan(np.deg2rad(spec.talus_deg)) * ctx.cell_size_m / ctx.height_scale_m
    )
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for _ in range(spec.iterations):
        diffs = [h - _shift(h, dy, dx) for dy, dx in neigh]
        excess = [np.maximum(d - talus, 0.0) for d in diffs]
        total = np.add.reduce(excess)
        moving = np.where(total > 0, spec.strength * 0.5 * np.maximum.reduce(excess), 0.0)
        share = np.divide(moving, total, out=np.zeros_like(total), where=total > 0)
        delta = np.zeros_like(h)
        for (dy, dx), e in zip(neigh, excess):
            give = e * share
            delta -= give
            # What leaves this cell arrives at the neighbour it flowed to.
            delta += _shift(give, -dy, -dx, fill=0.0)
        h += delta

    if spec.smooth_sigma_px > 0:
        h = ndimage.gaussian_filter(h, spec.smooth_sigma_px, mode="nearest")
    return (h - ctx.base).astype(np.float32)


_DISPATCH = {
    "peak": op_peak,
    "dune": op_dune,
    "terrace": op_terrace,
    "erosion": op_erosion,
}


def apply_operator(spec, ctx: OperatorContext) -> np.ndarray:
    """Evaluate one ``G_rj`` and return its unscaled delta."""
    try:
        fn = _DISPATCH[spec.kind]
    except KeyError:  # pragma: no cover -- schema keeps this unreachable
        raise ValueError(f"unknown terrain operator: {spec.kind!r}") from None
    return fn(spec, ctx)
