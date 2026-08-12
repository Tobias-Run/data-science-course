"""Topography fidelity of the image edit -- the sharpest metric in section 5.

Stage 3 renders the terrain, has an image model insert objects into that render,
and then back-projects rays through the *original* camera to place the results.
The whole construction assumes the edit left the terrain and the viewpoint
alone.  When the model instead nudges a ridge, shifts the horizon or subtly
re-frames the shot, every ray lands somewhere else and the placements are wrong
in a way no later step can detect.

So the edit is gated before it is used.

The asymmetry that makes this work
----------------------------------
An inserted object can only ever bring the surface *closer* to the camera: it
occludes terrain, it cannot reveal anything behind it.  So depth **decreasing**
is expected and allowed, while depth **increasing** means the terrain itself
moved.  That one-sided test separates legitimate edits from broken ones without
needing to know where the objects were placed.

Working with estimated depth
----------------------------
The "before" image can be given a true depth pass by Blender, but the edited
image only exists as pixels, so its depth has to be estimated -- and monocular
estimators are ambiguous up to scale and offset.  Comparing raw values would
therefore flag every edit.  ``align_depth`` fits a robust scale and shift on the
pixels that did not change, and the residual is what gets judged.

Known limitation
----------------
A camera dolly along its own optical axis shifts every depth by a constant, and
that is exactly what the alignment step is built to absorb -- so this gate
cannot see it.  It covers terrain deformation and any re-framing that changes
depth *structure* (panning, tilting, a moved horizon); a pure dolly has to be
prevented on the prompt side, by never asking the editor to change the
viewpoint.  ``test_a_pure_depth_offset_is_knowingly_undetectable`` pins the
limitation so it stays a known gap rather than becoming a surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopographyReport:
    accepted: bool
    reason: str
    raised_fraction: float
    changed_fraction: float
    median_residual: float
    p99_residual: float
    alignment: tuple[float, float] = (1.0, 0.0)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "raised_fraction": self.raised_fraction,
            "changed_fraction": self.changed_fraction,
            "median_residual": self.median_residual,
            "p99_residual": self.p99_residual,
            "alignment_scale": self.alignment[0],
            "alignment_shift": self.alignment[1],
            "notes": self.notes,
        }


def align_depth(
    before: np.ndarray, after: np.ndarray, trim: float = 0.25
) -> tuple[np.ndarray, tuple[float, float]]:
    """Fit ``after ~ a * before + b`` on the most consistent pixels.

    Monocular depth is defined only up to scale and offset, so an unaligned
    comparison measures the estimator's arbitrary normalisation rather than the
    scene.  The fit is trimmed: the pixels the edit legitimately changed are
    exactly the outliers, and including them would drag the alignment towards
    hiding the very thing being tested.
    """
    b = np.asarray(before, dtype=np.float64).ravel()
    a = np.asarray(after, dtype=np.float64).ravel()
    ok = np.isfinite(b) & np.isfinite(a)
    b, a = b[ok], a[ok]
    if len(b) < 16:
        return np.asarray(after, dtype=np.float64), (1.0, 0.0)

    scale, shift = 1.0, 0.0
    keep = np.ones(len(b), dtype=bool)
    for _ in range(5):
        A = np.stack([b[keep], np.ones(keep.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, a[keep], rcond=None)
        scale, shift = float(sol[0]), float(sol[1])
        resid = np.abs(a - (scale * b + shift))
        cutoff = np.quantile(resid, 1.0 - trim)
        keep = resid <= max(cutoff, 1e-12)
        if keep.sum() < 16:
            break

    aligned = (np.asarray(after, dtype=np.float64) - shift) / (scale if abs(scale) > 1e-12 else 1.0)
    return aligned, (scale, shift)


def compare_depth(
    before: np.ndarray,
    after: np.ndarray,
    tolerance: float = 0.02,
    max_raised: float = 0.02,
    max_changed: float = 0.45,
    align: bool = True,
) -> TopographyReport:
    """Judge whether an edit preserved the topography.

    ``tolerance`` is relative to the scene's depth range.  ``max_raised`` is the
    share of pixels allowed to have moved *away* from the camera -- the
    one-sided test -- and ``max_changed`` bounds how much of the frame the
    inserted objects may cover before the comparison stops being meaningful.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    if b.shape != a.shape:
        return TopographyReport(False, f"shape mismatch {b.shape} vs {a.shape}",
                                1.0, 1.0, np.inf, np.inf)

    alignment = (1.0, 0.0)
    if align:
        a, alignment = align_depth(b, a)

    span = float(np.nanmax(b) - np.nanmin(b))
    if not np.isfinite(span) or span <= 0:
        return TopographyReport(False, "reference depth map has no range", 1.0, 1.0,
                                np.inf, np.inf, alignment)
    tol_abs = tolerance * span

    diff = a - b
    finite = np.isfinite(diff)
    raised = float((diff > tol_abs)[finite].mean())
    changed = float((np.abs(diff) > tol_abs)[finite].mean())
    resid = np.abs(diff)[finite]
    median = float(np.median(resid) / span)
    p99 = float(np.percentile(resid, 99) / span)

    notes: list[str] = []
    if raised > max_raised:
        return TopographyReport(
            False,
            f"terrain moved away from the camera on {raised:.1%} of pixels "
            f"(limit {max_raised:.0%}) -- the edit changed the topography",
            raised, changed, median, p99, alignment, notes,
        )
    if changed > max_changed:
        return TopographyReport(
            False,
            f"{changed:.1%} of the frame changed depth (limit {max_changed:.0%}); "
            "the edit is too invasive to back-project through",
            raised, changed, median, p99, alignment, notes,
        )
    if changed <= 1e-6:
        notes.append("no depth change at all -- did the edit insert anything?")
    return TopographyReport(True, "topography preserved", raised, changed, median, p99,
                            alignment, notes)


# --------------------------------------------------------------- estimators


class TrueDepth:
    """Depth straight from Blender's Z pass -- exact, for the pre-edit render."""

    name = "blender_z_pass"

    def __call__(self, image_path: str) -> np.ndarray:  # pragma: no cover - I/O
        return np.load(image_path)


class ScriptedDepth:
    """Replays prepared maps, so the gate is testable without a depth model."""

    name = "scripted"

    def __init__(self, maps: list[np.ndarray]):
        self.maps = list(maps)

    def __call__(self, image_path: str) -> np.ndarray:
        if not self.maps:
            raise RuntimeError("ScriptedDepth ran out of prepared maps")
        return self.maps.pop(0)
