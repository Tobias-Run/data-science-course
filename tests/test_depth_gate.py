"""Tests for the topography-fidelity gate.

No depth model runs here.  Synthetic "before" maps are perturbed in the ways a
real image edit either legitimately does (adding an object) or must never do
(moving the terrain, re-framing the shot), and the gate has to tell them apart.
That is the whole point of building this before M4: the failure it guards
against is invisible in the edited image itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldclaw.placement.depth_gate import align_depth, compare_depth


def base_depth(h: int = 96, w: int = 128) -> np.ndarray:
    """A plausible terrain depth map: nearer at the bottom, ridged."""
    yy = np.linspace(1.0, 0.0, h)[:, None]
    xx = np.linspace(0.0, 1.0, w)[None, :]
    return (40.0 + 160.0 * yy + 12.0 * np.sin(6.0 * np.pi * xx) * yy).astype(np.float64)


def insert_object(depth: np.ndarray, box=(30, 40, 55, 75), depth_of_object=45.0) -> np.ndarray:
    """An object occludes terrain: depth in its footprint gets *smaller*."""
    out = depth.copy()
    r0, c0, r1, c1 = box
    out[r0:r1, c0:c1] = np.minimum(out[r0:r1, c0:c1], depth_of_object)
    return out


# ------------------------------------------------------------------ accepted


def test_unchanged_image_passes():
    d = base_depth()
    report = compare_depth(d, d.copy())
    assert report.accepted
    assert report.changed_fraction == pytest.approx(0.0)
    assert "insert anything" in " ".join(report.notes)


def test_inserted_object_passes():
    d = base_depth()
    report = compare_depth(d, insert_object(d))
    assert report.accepted, report.reason
    assert report.raised_fraction == pytest.approx(0.0)
    assert report.changed_fraction > 0.0


def test_several_inserted_objects_pass():
    d = base_depth()
    after = insert_object(d, (20, 10, 40, 30), 60.0)
    after = insert_object(after, (55, 70, 80, 95), 35.0)
    assert compare_depth(d, after).accepted


def test_mild_noise_passes():
    """A depth estimator is not bit-exact; the gate must survive its jitter."""
    d = base_depth()
    gen = np.random.default_rng(0)
    after = d + gen.normal(0.0, 0.3, d.shape)
    assert compare_depth(d, after).accepted


# ------------------------------------------------------------------ rejected


def test_terrain_pushed_away_is_rejected():
    """The one-sided test: an object can never move terrain *back*."""
    d = base_depth()
    after = d.copy()
    after[10:60, 20:100] += 25.0  # a ridge the model flattened / pushed away
    report = compare_depth(d, after)
    assert not report.accepted
    assert "moved away" in report.reason
    assert report.raised_fraction > 0.02


def test_laterally_reframed_shot_is_rejected():
    """Panning changes the depth structure, which alignment cannot absorb."""
    d = base_depth()
    report = compare_depth(d, np.roll(d, shift=9, axis=1))
    assert not report.accepted


def test_tilted_horizon_is_rejected():
    d = base_depth()
    h, w = d.shape
    tilt = np.linspace(-1.0, 1.0, w)[None, :] * np.linspace(0.0, 18.0, h)[:, None]
    assert not compare_depth(d, d + tilt).accepted


def test_a_pure_depth_offset_is_knowingly_undetectable():
    """Documented limitation, asserted so it cannot regress into a surprise.

    A camera dolly along its own optical axis shifts every depth by a constant.
    Monocular depth is only defined up to scale and offset, so that shift is
    indistinguishable from the estimator's own normalisation -- the alignment
    step absorbs it by construction.

    The gate therefore covers terrain deformation and any re-framing that
    changes depth *structure*; a pure dolly has to be prevented on the prompt
    side instead, by never asking the editor to change the viewpoint.
    """
    d = base_depth()
    report = compare_depth(d, d + 14.0)
    assert report.accepted
    assert report.alignment[1] == pytest.approx(14.0, abs=0.5)


def test_wholesale_repaint_is_rejected():
    d = base_depth()
    gen = np.random.default_rng(1)
    after = gen.uniform(d.min(), d.max(), d.shape)
    assert not compare_depth(d, after).accepted


def test_object_covering_most_of_the_frame_is_rejected():
    """Even a strictly-nearer edit is unusable if it hides the terrain."""
    d = base_depth()
    after = insert_object(d, (0, 0, 90, 120), 30.0)
    report = compare_depth(d, after)
    assert not report.accepted
    assert "too invasive" in report.reason


def test_shape_mismatch_is_rejected_not_crashed():
    assert not compare_depth(base_depth(), base_depth(48, 64)).accepted


# ----------------------------------------------------------------- alignment


def test_alignment_absorbs_scale_and_offset():
    """Monocular depth is defined up to scale and shift.

    Without alignment the gate would reject every edit whose depth came from an
    estimator rather than from Blender, which is every edited image there is.
    """
    d = base_depth()
    after = 2.5 * insert_object(d) + 17.0
    assert not compare_depth(d, after, align=False).accepted
    report = compare_depth(d, after, align=True)
    assert report.accepted, report.reason
    assert report.alignment[0] == pytest.approx(2.5, rel=0.05)


def test_alignment_does_not_hide_a_moved_terrain():
    """The fit must not absorb the very defect it is meant to expose.

    A trimmed fit is what makes this hold: the changed pixels are the outliers,
    so they are excluded from the fit instead of dragging it towards them.
    """
    d = base_depth()
    after = d.copy()
    after[5:70, 10:110] += 30.0
    after = 1.8 * after + 5.0  # plus an estimator's arbitrary normalisation
    report = compare_depth(d, after, align=True)
    assert not report.accepted
    assert "moved away" in report.reason


def test_align_depth_recovers_a_known_transform():
    d = base_depth()
    _, (scale, shift) = align_depth(d, 3.25 * d - 8.0)
    assert scale == pytest.approx(3.25, rel=1e-6)
    assert shift == pytest.approx(-8.0, abs=1e-6)


def test_gate_reports_numbers_not_just_a_verdict():
    """The report has to be diagnosable, since a rejected edit gets retried."""
    d = base_depth()
    report = compare_depth(d, insert_object(d))
    dumped = report.model_dump()
    for key in ("raised_fraction", "changed_fraction", "median_residual",
                "p99_residual", "alignment_scale"):
        assert key in dumped and np.isfinite(dumped[key])
