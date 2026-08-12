"""Automated acceptance checks.

The paper evaluates qualitatively, so "it got better" is only arguable if we
measure something ourselves.  These are the checks that apply to M1; the ones
the briefing lists for later milestones are declared here as ``SKIPPED`` with
the milestone that will implement them, so the report always shows the full
criterion list rather than quietly omitting what is not built yet.

Run with ``worldclaw check --run <id> --spec <spec>``; a failing check exits
non-zero, which is what makes this usable as a gate rather than a dashboard.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .layout import masks as masks_mod
from .schemas import TerrainSpec
from .terrain import heightfield as hf_mod


@dataclass
class Check:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


# --------------------------------------------------------------------------
# M1 checks
# --------------------------------------------------------------------------


def check_region_coverage(spec: TerrainSpec, rm: masks_mod.RegionMasks) -> Check:
    """Every region the spec declares must actually appear on the layout map.

    An unpainted region is silent: the height field is still produced, just
    without that landform.  This check exists because exactly that happened --
    a blob generator dropped both rock regions and the pipeline reported success.
    """
    cover = {n: float(rm.hard[i].mean()) for i, n in enumerate(rm.names)}
    empty = [n for n, c in cover.items() if c <= 0.0]
    return Check(
        "region_coverage",
        "fail" if empty else "pass",
        f"unpainted regions: {empty}" if empty else f"all {len(cover)} regions painted",
        {"coverage": cover, "unmatched_colour_fraction": rm.unmatched_fraction},
    )


def check_partition_of_unity(rm: masks_mod.RegionMasks, tol: float = 1e-4) -> Check:
    """The soft weights must sum to 1 everywhere.

    If they do not, the height field is no longer a convex combination of the
    regional profiles and every region border gains a step or a dip that no
    operator asked for.
    """
    err = float(np.abs(rm.weights.sum(axis=0) - 1.0).max())
    return Check(
        "partition_of_unity",
        "pass" if err <= tol else "fail",
        f"max |sum(m_tilde) - 1| = {err:.2e} (tol {tol:.0e})",
        {"max_error": err},
    )


def check_finite(hf: hf_mod.Heightfield) -> Check:
    bad = int((~np.isfinite(hf.height_m)).sum())
    return Check(
        "finite_heightfield",
        "pass" if bad == 0 else "fail",
        f"{bad} non-finite samples" if bad else "all samples finite",
        {"non_finite": bad},
    )


def check_slope_plausibility(hf: hf_mod.Heightfield, max_deg: float = 85.0) -> Check:
    """Near-vertical cells mean the raster is under-sampling a cliff.

    A height field cannot represent an overhang, so anything approaching 90
    degrees is a discretisation artefact rather than a landform, and it is what
    makes object placement on a slope fail later.
    """
    slope = hf.slope_deg()
    frac = float((slope > max_deg).mean())
    return Check(
        "slope_plausibility",
        "pass" if frac < 0.001 else "fail",
        f"{frac:.4%} of cells steeper than {max_deg}deg, p99 = {np.percentile(slope, 99):.1f}deg",
        {
            "fraction_over_max": frac,
            "p50_deg": float(np.percentile(slope, 50)),
            "p99_deg": float(np.percentile(slope, 99)),
            "max_deg": float(slope.max()),
        },
    )


def check_reproducibility(spec: TerrainSpec, layout_path: Path, reference: str) -> Check:
    """Same seed, same spec, same layout -> byte-identical height field."""
    rgb = masks_mod.load_layout(str(layout_path), spec.resolution)
    rm = masks_mod.extract_masks(spec, rgb)
    hf = hf_mod.build_heightfield(spec, rm)
    from .artifacts import sha256_bytes

    got = sha256_bytes(hf.height_m.tobytes())
    return Check(
        "reproducibility",
        "pass" if got == reference else "fail",
        f"rebuild sha256 {got[:12]} vs recorded {reference[:12]}",
        {"rebuilt": got, "recorded": reference},
    )


def check_heightmap_quantisation(hf: hf_mod.Heightfield, png_path: Path) -> Check:
    """The 16-bit export must round-trip within its own quantisation step.

    This is the raster that becomes an Unreal landscape in M5; a silent range or
    endianness mistake here would only show up as a mangled terrain much later.
    """
    q = np.asarray(Image.open(png_path), dtype=np.float64)
    lo, hi = float(hf.height_m.min()), float(hf.height_m.max())
    decoded = lo + q / 65535.0 * (hi - lo)
    err = float(np.abs(decoded - hf.height_m).max())
    step = (hi - lo) / 65535.0
    return Check(
        "heightmap_quantisation",
        "pass" if err <= step else "fail",
        f"max round-trip error {err:.4f} m (quantisation step {step:.4f} m)",
        {"max_error_m": err, "step_m": step},
    )


def check_stage_timing(manifest: dict) -> Check:
    """Cost and runtime per stage -- an acceptance criterion in its own right."""
    stages = {k: v.get("seconds", 0.0) for k, v in manifest.get("stages", {}).items()}
    return Check(
        "stage_timing",
        "pass",
        "  ".join(f"{k}={v:.2f}s" for k, v in stages.items()) or "no timings recorded",
        {"seconds": stages, "total_seconds": round(sum(stages.values()), 3)},
    )


# --------------------------------------------------------------------------
# Criteria that belong to later milestones
# --------------------------------------------------------------------------

_DEFERRED = {
    "scale_plausibility": "M4 -- needs reconstructed object heights per category",
    "topography_fidelity_of_edit": "M4 -- depth map before/after the image edit",
    "cost_per_scene": "M3 -- no model calls in the terrain stage yet",
}


def check_scatter_contact(scatter: dict, min_rate: float = 0.95) -> Check:
    """Contact rate of the scattered props.

    This is the paper's contact-rate criterion, measured against the terrain for
    M2's props.  M4 reuses the same measurement for reconstructed object meshes,
    so the number is comparable across milestones rather than being a
    scatter-specific invention.
    """
    stats = scatter.get("statistics", {})
    n = int(stats.get("count", 0))
    if n == 0:
        return Check("scatter_contact", "skipped", "no scatter specs in this terrain", stats)
    rate = float(stats.get("contact_rate", 0.0))
    return Check(
        "scatter_contact",
        "pass" if rate >= min_rate else "fail",
        f"{rate:.1%} of {n} props in contact (>= {min_rate:.0%} required), "
        f"max gap {stats.get('max_gap_m', 0):.3f} m, "
        f"max penetration {stats.get('max_penetration_m', 0):.3f} m",
        stats,
    )


def check_splat_partition(splat_dir: Path, tol: float = 2.0 / 255.0) -> Check:
    """The splat maps must still partition after 8-bit quantisation.

    The shader blends by these channels; if they no longer sum to one, the
    terrain gains bands that are darker or brighter than any region specifies.
    """
    from PIL import Image

    totals = None
    files = sorted(splat_dir.glob("splat_*.png"))
    if not files:
        return Check("splat_partition", "skipped", "no splat maps written")
    for p in files:
        a = np.asarray(Image.open(p), dtype=np.float32) / 255.0
        s = a.sum(axis=-1)
        totals = s if totals is None else totals + s
    err = float(np.abs(totals - 1.0).max())
    return Check(
        "splat_partition",
        "pass" if err <= tol else "fail",
        f"max |sum(channels) - 1| = {err:.4f} over {len(files)} map(s) (tol {tol:.4f})",
        {"max_error": err, "maps": len(files)},
    )


def run_all(run_dir: Path, spec: TerrainSpec, layout_path: Path) -> list[Check]:
    t0 = time.perf_counter()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    with np.load(run_dir / "terrain" / "weights.npz") as z:
        rm = masks_mod.RegionMasks(
            labels=z["labels"],
            hard=z["hard"],
            weights=z["weights"],
            names=[r.name for r in spec.regions],
            unmatched_fraction=json.loads(
                (run_dir / "terrain" / "masks.json").read_text(encoding="utf-8")
            )["unmatched_fraction"],
        )
    with np.load(run_dir / "terrain" / "heightfield.npz") as z:
        hf = hf_mod.Heightfield(height_m=z["height_m"], normalised=z["normalised"], spec=spec)

    art = json.loads((run_dir / "terrain_artifacts.json").read_text(encoding="utf-8"))
    scatter_path = run_dir / "terrain" / "scatter.json"
    scatter = json.loads(scatter_path.read_text(encoding="utf-8")) if scatter_path.exists() else {}

    checks = [
        check_region_coverage(spec, rm),
        check_partition_of_unity(rm),
        check_finite(hf),
        check_slope_plausibility(hf),
        check_heightmap_quantisation(hf, run_dir / "terrain" / "heightmap_16bit.png"),
        check_reproducibility(spec, layout_path, art["checksum"]),
        check_scatter_contact(scatter),
        check_splat_partition(run_dir / "terrain" / "splat"),
        check_stage_timing(manifest),
    ]
    checks += [Check(name, "skipped", why) for name, why in _DEFERRED.items()]
    checks.append(
        Check("check_runtime", "pass", f"checks took {time.perf_counter() - t0:.2f}s")
    )
    return checks


def format_report(checks: list[Check]) -> str:
    mark = {"pass": "PASS", "fail": "FAIL", "skipped": "skip"}
    lines = [f"  [{mark[c.status]}] {c.name:<28} {c.detail}" for c in checks]
    failed = [c.name for c in checks if c.status == "fail"]
    lines.append("")
    lines.append(f"  {sum(c.status == 'pass' for c in checks)} passed, "
                 f"{len(failed)} failed, {sum(c.status == 'skipped' for c in checks)} skipped")
    if failed:
        lines.append(f"  failing: {', '.join(failed)}")
    return "\n".join(lines)
