"""The global height field.

    H(x) = sum_r  m_tilde_r(x) * [ h_r + sum_k w_rk N_rk(x) + sum_j alpha_rj G_rj(x) ]

Operators see the regional base *including* its noise, which is what makes
terrace and erosion meaningful -- they modify a profile rather than invent one.
The regional profile is evaluated over the whole raster and only then weighted,
so a region's landform fades across its border instead of being clipped at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..layout.masks import RegionMasks
from ..schemas import RegionInstance, TerrainSpec
from .noise import noise_stack
from .operators import OperatorContext, apply_operator


@dataclass
class Heightfield:
    height_m: np.ndarray  # (H, W) float32, metres
    normalised: np.ndarray  # (H, W) float32, pre-scale units
    spec: TerrainSpec
    per_region: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def resolution(self) -> int:
        return self.height_m.shape[0]

    def slope_deg(self) -> np.ndarray:
        """Surface slope from the height gradient, in degrees."""
        gy, gx = np.gradient(self.height_m.astype(np.float32), self.spec.cell_size_m)
        return np.rad2deg(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    def normals(self) -> np.ndarray:
        """Unit surface normals, (H, W, 3), z up.

        Raster row 0 is the north edge, so the row index runs *against* world y:
        ``d/dy = -d/drow``.  Getting that sign wrong flips the lighting
        north-south and makes every hill read as a pit.
        """
        d_row, d_col = np.gradient(self.height_m.astype(np.float32), self.spec.cell_size_m)
        dz_dx, dz_dy = d_col, -d_row
        n = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], axis=-1)
        return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)

    def stats(self) -> dict[str, float]:
        s = self.slope_deg()
        return {
            "height_min_m": float(self.height_m.min()),
            "height_max_m": float(self.height_m.max()),
            "height_mean_m": float(self.height_m.mean()),
            "slope_mean_deg": float(s.mean()),
            "slope_p99_deg": float(np.percentile(s, 99)),
        }


def regional_profile(
    spec: TerrainSpec, masks: RegionMasks, region_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one region's bracket term. Returns (profile, base_before_ops)."""
    region = spec.regions[region_index]
    shape = masks.labels.shape

    base = np.full(shape, np.float32(region.base_height), dtype=np.float32)
    if region.noise:
        base = base + noise_stack(shape, region.noise, spec.seed, region.name)

    profile = base.copy()
    for j, op in enumerate(region.operators):
        ctx = OperatorContext(
            base=profile,
            mask=masks.hard[region_index],
            cell_size_m=spec.cell_size_m,
            height_scale_m=spec.height_scale_m,
            world_size_m=spec.world_size_m,
            root_seed=spec.seed,
            region=region.name,
            index=j,
        )
        profile = profile + op.alpha * apply_operator(op, ctx)
    return profile.astype(np.float32), base


def build_heightfield(spec: TerrainSpec, masks: RegionMasks, keep_regions: bool = False) -> Heightfield:
    """Assemble ``H`` from the regional profiles and the shared soft weights."""
    if masks.n_regions != len(spec.regions):
        raise ValueError("mask stack and spec regions disagree")

    total = np.zeros(masks.labels.shape, dtype=np.float32)
    per_region: dict[str, np.ndarray] = {}
    for i, region in enumerate(spec.regions):
        profile, _ = regional_profile(spec, masks, i)
        total += masks.weights[i] * profile
        if keep_regions:
            per_region[region.name] = profile

    return Heightfield(
        height_m=(total * spec.height_scale_m).astype(np.float32),
        normalised=total,
        spec=spec,
        per_region=per_region,
    )


def region_instances(
    spec: TerrainSpec, masks: RegionMasks, hf: Heightfield, min_area_px: int = 64
) -> list[RegionInstance]:
    """Describe every painted patch: where it is, how high, how steep.

    Stage 3 picks its working regions from this list, and the slope statistics
    are what make "put a camp here" a decidable question before any model runs.
    """
    slope = hf.slope_deg()
    cell = spec.cell_size_m
    out: list[RegionInstance] = []

    for i, region in enumerate(spec.regions):
        lab, n = ndimage.label(masks.hard[i])
        for k in range(1, n + 1):
            comp = lab == k
            area = int(comp.sum())
            if area < min_area_px:
                continue
            ys, xs = np.nonzero(comp)
            cy, cx = float(ys.mean()), float(xs.mean())
            h_vals = hf.height_m[comp]
            out.append(
                RegionInstance(
                    region=region.name,
                    index=k - 1,
                    area_px=area,
                    area_m2=area * cell * cell,
                    centroid_px=(cx, cy),
                    # World frame: x east, y north, origin at the raster centre.
                    centroid_m=(
                        (cx - masks.labels.shape[1] / 2) * cell,
                        (masks.labels.shape[0] / 2 - cy) * cell,
                    ),
                    bbox_px=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    mean_height_m=float(h_vals.mean()),
                    min_height_m=float(h_vals.min()),
                    max_height_m=float(h_vals.max()),
                    mean_slope_deg=float(slope[comp].mean()),
                )
            )
    out.sort(key=lambda r: r.area_px, reverse=True)
    return out
