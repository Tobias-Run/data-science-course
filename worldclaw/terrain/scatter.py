"""Asset scattering: where props go, and whether they actually touch the ground.

Sampling follows the briefing: candidate positions drawn inside the region
masks by affinity and target density, filtered by local height, slope and
normal, then scale and orientation adapted to the surface.

The contact test at the end is not decoration.  M2's acceptance criterion
("rocks and vegetation without floating or interpenetration") and the paper's
contact-rate metric for generated objects are the same measurement, so it is
implemented once here, against the terrain, and M4 reuses it for reconstructed
meshes instead of reinventing it under time pressure.
"""

from __future__ import annotations

import numpy as np

from ..rng import rng as make_rng
from ..schemas import ScatterInstance, ScatterSpec, TerrainSpec
from .frame import raster_from_world, sample_bilinear, sample_height_m, world_from_raster
from .heightfield import Heightfield


def _footprint_offsets(radius_m: float, rings: int = 3, per_ring: int = 8) -> np.ndarray:
    """Sample offsets over a disc: centre plus concentric rings.

    A disc rather than a bounding box because a prop's contact is decided by its
    footprint, and a box would test corners that hang over nothing.
    """
    pts = [(0.0, 0.0)]
    for i in range(1, rings + 1):
        r = radius_m * i / rings
        for k in range(per_ring):
            a = 2.0 * np.pi * k / per_ring + (0.5 * i)
            pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts, dtype=np.float64)


def _footprint_residuals(
    height_m: np.ndarray,
    cell_size_m: float,
    x: float,
    y: float,
    radius_m: float,
    axis: np.ndarray,
) -> np.ndarray:
    """Terrain heights under the footprint, relative to the prop's base *plane*.

    The base is a disc perpendicular to ``axis``, not a horizontal slab: a prop
    tilted onto the surface normal has a tilted base, and measuring it against a
    horizontal plane reports interpenetration that does not exist.  Subtracting
    the plane's own rise at each sample leaves a residual that is zero exactly
    when the base lies flush on the ground.
    """
    off = _footprint_offsets(radius_m)
    hs = np.asarray(
        sample_height_m(height_m, cell_size_m, x + off[:, 0], y + off[:, 1]), dtype=np.float64
    )
    az = max(float(axis[2]), 1e-6)
    plane_rise = -(float(axis[0]) * off[:, 0] + float(axis[1]) * off[:, 1]) / az
    return hs - plane_rise


def contact_report(
    height_m: np.ndarray,
    cell_size_m: float,
    x: float,
    y: float,
    base_z: float,
    radius_m: float,
    tolerance_m: float,
    axis: np.ndarray,
) -> tuple[float, float, float]:
    """How well a prop of the given footprint meets the terrain.

    Returns ``(contact_ratio, gap_m, penetration_m)``:

    * ``contact_ratio`` -- share of footprint samples whose terrain height is
      within ``tolerance_m`` of the base plane. This is the quantity the paper's
      contact search drives to a threshold.
    * ``gap_m`` -- largest distance the base floats above the terrain.
    * ``penetration_m`` -- largest distance the terrain pokes above the base.

    Both failure modes are reported separately because they need opposite fixes:
    a gap means lower or embed the prop, penetration means the footprint is too
    large for the local relief.
    """
    delta = _footprint_residuals(height_m, cell_size_m, x, y, radius_m, axis) - base_z
    contact = float((np.abs(delta) <= tolerance_m).mean())
    return contact, float(max(-delta.min(), 0.0)), float(max(delta.max(), 0.0))


def resolve_base_height(
    height_m: np.ndarray,
    cell_size_m: float,
    x: float,
    y: float,
    radius_m: float,
    embed: float,
    axis: np.ndarray,
) -> float:
    """Choose the base height for a prop standing on sloped ground.

    Sitting a prop at the centre height leaves it floating on the downhill side
    of any residual slope.  Anchoring low in the footprint's residual
    distribution and embedding a fraction of the local relief trades a little
    burial for contact all round -- the same trade the paper's terrain
    refinement makes when it partially embeds an object.
    """
    res = _footprint_residuals(height_m, cell_size_m, x, y, radius_m, axis)
    relief = float(res.max() - res.min())
    return float(np.percentile(res, 30.0) - embed * relief)


def scatter_region(
    spec: TerrainSpec,
    hf: Heightfield,
    weights_r: np.ndarray,
    region_name: str,
    scatter: ScatterSpec,
    slope_deg: np.ndarray,
    normals: np.ndarray,
) -> list[ScatterInstance]:
    """Sample one asset class over one region."""
    shape = hf.height_m.shape
    cell = spec.cell_size_m
    gen = make_rng(spec.seed, "scatter", region_name, scatter.asset_class)

    # Target count from density over the region's *effective* area: the soft
    # weights are the affinity field, so their sum is the area that actually
    # belongs to this region, borders included at partial strength.
    area_km2 = float(weights_r.sum()) * (cell * cell) / 1e6
    target = int(round(scatter.density_per_km2 * area_km2))
    if target <= 0:
        return []

    # Oversample, then filter: rejection is cheaper than trying to sample the
    # admissible set directly, and the filters are what enforce the spec.
    n_draw = min(target * 8, 400_000)
    flat = weights_r.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        return []
    idx = gen.choice(flat.size, size=n_draw, p=flat / total, replace=True)
    rows, cols = np.unravel_index(idx, shape)
    # Jitter inside the cell so props do not sit on a visible lattice.
    rows = rows + gen.uniform(-0.5, 0.5, n_draw)
    cols = cols + gen.uniform(-0.5, 0.5, n_draw)
    xs, ys = world_from_raster(shape, cell, rows, cols)

    keep = np.ones(n_draw, dtype=bool)
    keep &= np.asarray(sample_bilinear(slope_deg, rows, cols)) <= scatter.max_slope_deg
    heights = np.asarray(sample_height_m(hf.height_m, cell, xs, ys))
    if scatter.height_range_m is not None:
        lo, hi = scatter.height_range_m
        keep &= (heights >= lo) & (heights <= hi)
    xs, ys, rows, cols = xs[keep], ys[keep], rows[keep], cols[keep]
    if len(xs) == 0:
        return []

    # Minimum spacing on a hash grid: props that interpenetrate each other read
    # as a clump however well each one contacts the ground.
    order = gen.permutation(len(xs))
    spacing = max(scatter.min_spacing_m, 1e-6)
    taken: dict[tuple[int, int], list[int]] = {}
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= target:
            break
        gx, gy = int(xs[i] // spacing), int(ys[i] // spacing)
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in taken.get((gx + dx, gy + dy), ()):
                    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < spacing * spacing:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if not near:
            taken.setdefault((gx, gy), []).append(i)
            chosen.append(i)

    out: list[ScatterInstance] = []
    for n, i in enumerate(chosen):
        scale = float(gen.uniform(*scatter.scale_range))
        radius = scatter.footprint_radius_m * scale

        nrm = np.asarray(sample_bilinear(normals, rows[i], cols[i]), dtype=np.float64)
        nrm /= max(float(np.linalg.norm(nrm)), 1e-9)
        # Blend the surface normal towards world up; align_to_normal = 0 stands
        # the prop upright, 1 lays it flush with the slope.  Resolved before the
        # base height, because the base plane is perpendicular to this axis and
        # the contact geometry depends on it.
        up = np.array([0.0, 0.0, 1.0])
        axis = up + scatter.align_to_normal * (nrm - up)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)

        base_z = resolve_base_height(
            hf.height_m, cell, xs[i], ys[i], radius, scatter.embed, axis
        )
        contact, gap, pen = contact_report(
            hf.height_m, cell, xs[i], ys[i], base_z, radius, scatter.contact_tolerance_m, axis
        )
        out.append(
            ScatterInstance(
                asset_class=scatter.asset_class,
                region=region_name,
                index=n,
                position_m=(float(xs[i]), float(ys[i]), base_z),
                normal=(float(axis[0]), float(axis[1]), float(axis[2])),
                yaw_deg=float(gen.uniform(0.0, 360.0)),
                scale=scale,
                footprint_radius_m=radius,
                contact_ratio=contact,
                gap_m=gap,
                penetration_m=pen,
            )
        )
    return out


def scatter_scene(
    spec: TerrainSpec, hf: Heightfield, weights: np.ndarray
) -> list[ScatterInstance]:
    """Run every region's scatter specs. Slope and normals are computed once."""
    slope = hf.slope_deg()
    normals = hf.normals()
    out: list[ScatterInstance] = []
    for i, region in enumerate(spec.regions):
        for s in region.scatter:
            out.extend(
                scatter_region(spec, hf, weights[i], region.name, s, slope, normals)
            )
    return out


def contact_statistics(instances: list[ScatterInstance], threshold: float = 0.6) -> dict:
    """Aggregate contact quality -- the M2 gate and the M4 metric in one shape."""
    if not instances:
        return {"count": 0, "contact_rate": 1.0, "mean_contact_ratio": 1.0,
                "max_gap_m": 0.0, "max_penetration_m": 0.0}
    ratios = np.array([i.contact_ratio for i in instances])
    return {
        "count": len(instances),
        "contact_rate": float((ratios >= threshold).mean()),
        "mean_contact_ratio": float(ratios.mean()),
        "max_gap_m": float(max(i.gap_m for i in instances)),
        "max_penetration_m": float(max(i.penetration_m for i in instances)),
        "threshold": threshold,
    }
