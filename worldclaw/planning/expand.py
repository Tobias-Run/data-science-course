"""ScenePlan -> TerrainSpec.

The planner states character; this module turns character into numbers.  Keeping
the mapping here rather than in the prompt means the numbers are reviewable,
testable and identical across models -- swapping the local LLM changes the
scene's *design*, never its numerical sanity.
"""

from __future__ import annotations

from ..layout.palette import PALETTE
from ..schemas import (
    DuneOperator,
    ErosionOperator,
    MaterialSpec,
    NoiseComponent,
    PeakOperator,
    RegionSpec,
    ScatterSpec,
    TerraceOperator,
    TerrainSpec,
)
from .schema import RegionPlan, ScenePlan

# Base colour and roughness per category, in linear RGB.
_MATERIAL = {
    "water": ((0.06, 0.13, 0.20), 0.25),
    "basin": ((0.42, 0.40, 0.33), 0.88),
    "plain": ((0.28, 0.38, 0.18), 0.90),
    "sand": ((0.71, 0.63, 0.47), 0.90),
    "dune": ((0.78, 0.62, 0.36), 0.75),
    "slope": ((0.48, 0.34, 0.24), 0.95),
    "forest": ((0.13, 0.26, 0.13), 0.93),
    "plateau": ((0.62, 0.55, 0.44), 0.92),
    "rock": ((0.38, 0.37, 0.36), 0.85),
    "peak": ((0.82, 0.82, 0.84), 0.80),
}

# Roughness of the base noise stack per category, as (frequency, weight, octaves).
_BASE_NOISE = {
    "water": [(5.0, 0.004, 2)],
    "basin": [(6.0, 0.022, 4), (28.0, 0.005, 2)],
    "plain": [(4.0, 0.030, 4), (20.0, 0.008, 2)],
    "sand": [(4.0, 0.025, 3), (26.0, 0.006, 2)],
    "dune": [(2.5, 0.060, 3)],
    "slope": [(6.0, 0.070, 5), (20.0, 0.020, 3)],
    "forest": [(5.0, 0.040, 4), (18.0, 0.010, 3)],
    "plateau": [(3.0, 0.065, 5), (13.0, 0.018, 4)],
    "rock": [(5.0, 0.110, 5)],
    "peak": [(4.0, 0.130, 5)],
}

_RIDGED = {"slope", "rock", "peak"}

_DEFAULT_SCATTER = {
    "boulder": dict(density_per_km2=900.0, max_slope_deg=24.0, footprint_radius_m=1.1,
                    min_spacing_m=6.0, scale_range=(0.7, 1.6)),
    "rock": dict(density_per_km2=1400.0, max_slope_deg=26.0, footprint_radius_m=0.7,
                 min_spacing_m=4.0, scale_range=(0.6, 1.3)),
    "pebble": dict(density_per_km2=5000.0, max_slope_deg=16.0, footprint_radius_m=0.3,
                   min_spacing_m=1.6, scale_range=(0.5, 1.1)),
    "shrub": dict(density_per_km2=2400.0, max_slope_deg=20.0, footprint_radius_m=0.6,
                  min_spacing_m=3.0, scale_range=(0.7, 1.4)),
    "grass": dict(density_per_km2=6000.0, max_slope_deg=22.0, footprint_radius_m=0.4,
                  min_spacing_m=1.8, scale_range=(0.6, 1.2)),
    "tree": dict(density_per_km2=1800.0, max_slope_deg=25.0, footprint_radius_m=1.2,
                 min_spacing_m=7.0, scale_range=(0.75, 1.5)),
}


def _operators_for(region: RegionPlan, relief_m: float) -> list:
    """Character words -> geomorphological operators."""
    ops: list = []
    chars = set(region.character)

    if "peaked" in chars or region.category == "peak":
        ops.append(PeakOperator(alpha=0.30, count=3, radius=0.12, sharpness=2.2,
                                ridge_amount=0.45))
    if "dunes" in chars or region.category == "dune":
        ops.append(DuneOperator(alpha=0.22, wavelength=0.075, direction_deg=28.0,
                                asymmetry=0.62, crest_sharpness=1.7, meander=0.4))
        ops.append(DuneOperator(alpha=0.05, wavelength=0.021, direction_deg=47.0,
                                asymmetry=0.4, crest_sharpness=1.2, meander=0.6))
    if "terraced" in chars:
        # Steps are counted per normalised unit, so bench height in metres is
        # relief/steps: pick the step count from the relief to keep benches
        # around 12 m whatever the scene's scale.
        steps = max(4.0, min(50.0, relief_m / 12.0))
        ops.append(TerraceOperator(alpha=0.7, steps=steps, sharpness=0.72, warp=0.035))

    talus = {"rugged": 44.0, "eroded": 32.0, "smooth": 24.0}
    iterations = 20
    angle = 38.0
    for word, deg in talus.items():
        if word in chars:
            angle = deg
            iterations = 55 if word == "eroded" else 25
    if "flat" in chars:
        angle, iterations = 18.0, 15
    ops.append(ErosionOperator(alpha=1.0, iterations=iterations, talus_deg=angle,
                               strength=0.5,
                               smooth_sigma_px=0.6 if "smooth" in chars else 0.0))
    return ops


def _noise_for(region: RegionPlan) -> list[NoiseComponent]:
    bands = _BASE_NOISE.get(region.category, _BASE_NOISE["plain"])
    chars = set(region.character)
    gain = 1.0
    if "smooth" in chars or "flat" in chars:
        gain = 0.45
    elif "rugged" in chars:
        gain = 1.5
    out = []
    for i, (freq, weight, octaves) in enumerate(bands):
        kind = "ridged" if (i == 0 and region.category in _RIDGED) else "perlin"
        out.append(
            NoiseComponent(kind=kind, frequency=freq, weight=weight * gain, octaves=octaves)
        )
    return out


def _scatter_for(region: RegionPlan) -> list[ScatterSpec]:
    out = []
    for name in region.scatter:
        key = name.strip().lower()
        preset = _DEFAULT_SCATTER.get(key)
        if preset is None:
            # Unknown class: still scatter it, conservatively, rather than
            # dropping it silently -- the planner asked for it on purpose.
            preset = dict(density_per_km2=800.0, max_slope_deg=22.0,
                          footprint_radius_m=0.8, min_spacing_m=4.0)
        out.append(ScatterSpec(asset_class=key, **preset))
    return out


def expand(plan: ScenePlan, seed: int = 0, resolution: int = 768) -> TerrainSpec:
    """Turn a ScenePlan into a fully specified TerrainSpec."""
    if not plan.regions:
        raise ValueError("plan has no regions")

    # Elevations are planned on a 0..1 scale; the spec wants normalised height
    # units against height_scale_m, so relief maps directly onto that scale.
    height_scale = max(plan.relief_m, 1.0)

    used_colors: set[str] = set()
    regions: list[RegionSpec] = []
    for rp in plan.regions:
        color = PALETTE.get(rp.category, "#808080")
        if color in used_colors:
            # Two regions of the same category would collide on the layout map;
            # nudge the colour so quantisation can still tell them apart.
            r, g, b = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
            color = f"#{min(r + 24, 255):02x}{max(g - 20, 0):02x}{min(b + 16, 255):02x}"
        used_colors.add(color)

        base_rgb, roughness = _MATERIAL.get(rp.category, ((0.5, 0.5, 0.5), 0.9))
        regions.append(
            RegionSpec(
                name=rp.name.strip().lower().replace(" ", "_") or rp.category,
                color=color,
                base_height=float(rp.elevation),
                blend_width_px=6.0 if rp.category in _RIDGED else 10.0,
                noise=_noise_for(rp),
                operators=_operators_for(rp, plan.relief_m),
                material=MaterialSpec(base_color=base_rgb, roughness=roughness),
                scatter=_scatter_for(rp),
            )
        )

    return TerrainSpec(
        name=plan.name.strip().lower().replace(" ", "-") or "scene",
        seed=seed,
        world_size_m=plan.world_size_m,
        height_scale_m=height_scale,
        resolution=resolution,
        blend_width_px=8.0,
        despeckle_px=2,  # generated layout maps are noisier than painted ones
        layout_map=f"layouts/{plan.name.strip().lower().replace(' ', '-')}.png",
        regions=regions,
    )
