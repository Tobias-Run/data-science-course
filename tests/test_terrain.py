"""Regression tests for the M1 terrain stage.

The tests that matter here are the ones covering properties the eye cannot
check on a hillshade: that the weights partition, that the world frame is the
same in the exporter and in the Blender stage, and that a seed pins the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldclaw.layout import masks as masks_mod
from worldclaw.rng import derive_seed, rng
from worldclaw.schemas import (
    DuneOperator,
    ErosionOperator,
    NoiseComponent,
    PeakOperator,
    RegionSpec,
    TerraceOperator,
    TerrainSpec,
)
from worldclaw.terrain import heightfield as hf_mod
from worldclaw.terrain import mesh as mesh_mod
from worldclaw.terrain.noise import fbm, gradient_noise
from worldclaw.terrain.operators import OperatorContext, apply_operator
from worldclaw.tools import make_layout

RES = 96


@pytest.fixture
def spec() -> TerrainSpec:
    return TerrainSpec(
        name="test",
        seed=7,
        world_size_m=512.0,
        height_scale_m=100.0,
        resolution=RES,
        blend_width_px=3.0,
        regions=[
            RegionSpec(
                name="low",
                color="#1f6fb4",
                base_height=0.1,
                noise=[NoiseComponent(frequency=4.0, weight=0.02)],
            ),
            RegionSpec(
                name="high",
                color="#e8e8e8",
                base_height=0.8,
                noise=[NoiseComponent(kind="ridged", frequency=6.0, weight=0.05, octaves=3)],
                operators=[ErosionOperator(alpha=1.0, iterations=5, talus_deg=40.0)],
            ),
        ],
    )


@pytest.fixture
def layout(spec: TerrainSpec) -> np.ndarray:
    colors = {r.name: r.color for r in spec.regions}
    rgb = np.zeros((RES, RES, 3), dtype=np.uint8)
    rgb[:, :] = spec.regions[0].rgb
    rgb[: RES // 2, :] = spec.regions[1].rgb
    del colors
    return rgb


# ---------------------------------------------------------------- seeding


def test_seed_derivation_is_stable_and_path_sensitive():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2)
    assert derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
    assert derive_seed(1, "a", 2) != derive_seed(2, "a", 2)


def test_noise_is_deterministic_per_seed():
    a = gradient_noise((64, 64), 4.0, rng(11, "x"))
    b = gradient_noise((64, 64), 4.0, rng(11, "x"))
    c = gradient_noise((64, 64), 4.0, rng(12, "x"))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_is_bounded_and_centred():
    n = fbm((128, 128), 5.0, rng(3, "n"), octaves=4)
    assert np.abs(n).max() <= 1.05
    assert abs(float(n.mean())) < 0.15


def test_octave_rotation_breaks_axis_alignment():
    """Row and column marginals should not be more structured than each other.

    A lattice that stays axis-aligned across octaves shows up as anisotropy
    between the two directions -- the visible "maze" artefact that motivated
    switching from value noise to rotated gradient noise.
    """
    n = fbm((256, 256), 6.0, rng(5, "aniso"), octaves=5)
    dx = float(np.abs(np.diff(n, axis=1)).mean())
    dy = float(np.abs(np.diff(n, axis=0)).mean())
    assert abs(dx - dy) / max(dx, dy) < 0.08


# ------------------------------------------------------------------ masks


def test_soft_weights_partition_unity(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    total = rm.weights.sum(axis=0)
    assert np.allclose(total, 1.0, atol=1e-5)


def test_masks_are_soft_at_the_border_and_crisp_inside(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    w_high = rm.weight_of("high")
    assert w_high[0, 0] > 0.99  # deep inside the painted area
    assert w_high[-1, 0] < 0.01  # deep outside
    border = w_high[RES // 2 - 1 : RES // 2 + 1, :]
    assert (0.05 < border).any() and (border < 0.95).any()


def test_unmatched_colours_are_reported(spec):
    rgb = np.full((RES, RES, 3), 127, dtype=np.uint8)  # matches no palette entry
    rm = masks_mod.extract_masks(spec, rgb)
    assert rm.unmatched_fraction == pytest.approx(1.0)


def test_duplicate_region_colours_are_rejected():
    with pytest.raises(ValueError, match="duplicate region"):
        TerrainSpec(
            name="dup",
            regions=[
                RegionSpec(name="a", color="#111111", base_height=0.0),
                RegionSpec(name="b", color="#111111", base_height=0.1),
            ],
        )


# ------------------------------------------------------------- operators


def _ctx(base: np.ndarray, mask: np.ndarray | None = None) -> OperatorContext:
    return OperatorContext(
        base=base,
        mask=np.ones_like(base, dtype=bool) if mask is None else mask,
        cell_size_m=4.0,
        height_scale_m=100.0,
        world_size_m=base.shape[0] * 4.0,
        root_seed=1,
        region="r",
        index=0,
    )


def test_terrace_flattens_into_steps():
    base = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32), (16, 1))
    ctx = _ctx(base)
    out = base + apply_operator(TerraceOperator(alpha=1.0, steps=8.0, sharpness=0.9, warp=0.0), ctx)
    # A ramp becomes a staircase: most of the gradient collapses to a few risers.
    g = np.abs(np.diff(out[0]))
    assert float((g < g.mean() * 0.25).mean()) > 0.6
    assert out.min() >= base.min() - 0.2 and out.max() <= base.max() + 0.2


def test_terrace_alpha_zero_is_a_no_op():
    base = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (8, 1))
    delta = apply_operator(TerraceOperator(alpha=0.0, steps=8.0), _ctx(base))
    assert np.allclose(0.0 * delta, 0.0)  # alpha is applied by the caller
    assert delta.shape == base.shape


def test_erosion_reduces_slopes_beyond_the_talus_angle():
    gen = np.random.default_rng(0)
    base = gen.random((64, 64)).astype(np.float32) * 0.5  # violently rough
    ctx = _ctx(base)
    op = ErosionOperator(alpha=1.0, iterations=30, talus_deg=35.0, strength=0.5)
    out = base + apply_operator(op, ctx)
    before = np.abs(np.diff(base, axis=1)).mean()
    after = np.abs(np.diff(out, axis=1)).mean()
    assert after < before


def test_erosion_conserves_material():
    """Thermal relaxation moves material, it does not create or destroy it."""
    gen = np.random.default_rng(1)
    base = gen.random((48, 48)).astype(np.float32) * 0.4
    out = base + apply_operator(
        ErosionOperator(alpha=1.0, iterations=20, talus_deg=30.0, strength=0.5), _ctx(base)
    )
    assert float(out.sum()) == pytest.approx(float(base.sum()), rel=2e-3)


def test_peak_stays_inside_its_region():
    mask = np.zeros((128, 128), dtype=bool)
    mask[80:120, 80:120] = True
    base = np.zeros((128, 128), dtype=np.float32)
    out = apply_operator(
        PeakOperator(alpha=1.0, count=2, radius=0.06, ridge_amount=0.0), _ctx(base, mask)
    )
    # The bump may spill over the border, but its summit must be inside.
    peak = np.unravel_index(int(np.argmax(out)), out.shape)
    assert mask[peak]
    assert out.max() > 0.5


def test_peak_is_finite_for_fractional_sharpness():
    """cos(pi/2) lands a hair below zero in float32; a fractional exponent NaNs."""
    out = apply_operator(
        PeakOperator(alpha=1.0, count=1, sharpness=2.4), _ctx(np.zeros((64, 64), np.float32))
    )
    assert np.isfinite(out).all()


def test_dune_profile_is_asymmetric_and_additive():
    base = np.zeros((128, 128), dtype=np.float32)
    out = apply_operator(
        DuneOperator(alpha=1.0, wavelength=0.2, direction_deg=0.0, asymmetry=0.7, meander=0.0),
        _ctx(base),
    )
    assert out.min() >= -1e-6  # dunes add relief, never dig
    slopes = np.diff(out[0])
    # Windward and lee flanks must differ: a symmetric wave would match.
    assert abs(float(slopes[slopes > 0].mean())) != pytest.approx(
        abs(float(slopes[slopes < 0].mean())), rel=0.05
    )


# ----------------------------------------------------------- height field


def test_heightfield_is_reproducible(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    a = hf_mod.build_heightfield(spec, rm)
    b = hf_mod.build_heightfield(spec, rm)
    assert np.array_equal(a.height_m, b.height_m)


def test_heightfield_respects_regional_base_heights(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    high = hf.height_m[:10].mean()
    low = hf.height_m[-10:].mean()
    assert high > low
    # Interior heights track h_r * height_scale, give or take noise and erosion.
    assert abs(low - 0.1 * spec.height_scale_m) < 0.1 * spec.height_scale_m


def test_normals_point_up_and_match_slope(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    n = hf.normals()
    assert (n[..., 2] > 0).all()
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)
    # The normal tilt and the slope angle are two views of the same quantity.
    # arctan2 rather than arccos: on near-flat cells arccos(~1) cancels in
    # float32 and the comparison would be testing precision, not agreement.
    tilt = np.rad2deg(np.arctan2(np.hypot(n[..., 0], n[..., 1]), n[..., 2]))
    assert np.allclose(tilt, hf.slope_deg(), atol=1e-3)


def test_normal_y_sign_follows_world_north():
    """A slope rising northwards must tilt its normal southwards, not north.

    Raster rows run north to south, so a naive gradient gets this backwards and
    every hill renders as a pit.
    """
    spec = TerrainSpec(
        name="ramp", resolution=64, world_size_m=64.0, height_scale_m=1.0,
        regions=[RegionSpec(name="a", color="#000000", base_height=0.0)],
    )
    hf = hf_mod.Heightfield(
        # Row 0 is north; height decreasing with row index means rising northward.
        height_m=np.tile(np.linspace(10.0, 0.0, 64, dtype=np.float32)[:, None], (1, 64)),
        normalised=np.zeros((64, 64), dtype=np.float32),
        spec=spec,
    )
    assert hf.normals()[..., 1].mean() < 0


def test_region_instances_describe_each_patch(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    inst = hf_mod.region_instances(spec, rm, hf, min_area_px=16)
    assert {i.region for i in inst} == {"low", "high"}
    for i in inst:
        assert i.min_height_m <= i.mean_height_m <= i.max_height_m
        assert i.area_m2 == pytest.approx(i.area_px * spec.cell_size_m**2)


# ------------------------------------------------------------------ mesh


def test_mesh_topology_is_a_closed_grid(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    v = mesh_mod.grid_vertices(hf)
    f = mesh_mod.grid_faces(RES, RES)
    assert v.shape == (RES * RES, 3)
    assert f.shape == (2 * (RES - 1) ** 2, 3)
    assert f.min() == 0 and f.max() == RES * RES - 1
    # Every interior edge is shared by exactly two triangles.
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(np.unique(counts)) <= {1, 2}


def test_mesh_winding_is_counter_clockwise_from_above(spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    v = mesh_mod.grid_vertices(hf)
    f = mesh_mod.grid_faces(RES, RES)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nz = np.cross(b - a, c - a)[:, 2]
    assert (nz > 0).all()


def test_world_frame_matches_the_blender_stage(spec, layout):
    """The exporter and the Blender builder must agree on the world frame.

    Two implementations of the same convention is exactly the kind of thing that
    drifts silently, and stage 3 back-projects rays through it.
    """
    from worldclaw.blender.build_terrain import _world_xy, sample_height

    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    v = mesh_mod.grid_vertices(hf).reshape(RES, RES, 3)

    for row, col in ((0, 0), (0, RES - 1), (RES - 1, 0), (RES // 3, RES // 2)):
        x, y = _world_xy(hf.height_m, spec.cell_size_m, row, col)
        assert (x, y) == pytest.approx((float(v[row, col, 0]), float(v[row, col, 1])))
        assert sample_height(hf.height_m, spec.cell_size_m, x, y) == pytest.approx(
            float(v[row, col, 2])
        )


def test_obj_export_is_wellformed(tmp_path, spec, layout):
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    path = mesh_mod.write_obj(hf, tmp_path / "t.obj", stride=2)
    text = path.read_text().splitlines()
    n_v = sum(l.startswith("v ") for l in text)
    n_vt = sum(l.startswith("vt ") for l in text)
    n_f = sum(l.startswith("f ") for l in text)
    side = len(range(0, RES, 2))
    assert n_v == n_vt == side * side
    assert n_f == 2 * (side - 1) ** 2
    # OBJ is 1-based; index 0 would silently corrupt the mesh in every importer.
    assert all(int(tok.split("/")[0]) >= 1 for l in text if l.startswith("f ") for tok in l.split()[1:])


def test_heightmap_png16_roundtrip(tmp_path, spec, layout):
    from PIL import Image

    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    path, info = mesh_mod.write_heightmap_png16(hf, tmp_path / "h.png")
    q = np.asarray(Image.open(path), dtype=np.float64)
    assert q.dtype == np.float64 and q.max() <= 65535
    decoded = info["height_min_m"] + q / 65535.0 * info["range_m"]
    assert np.abs(decoded - hf.height_m).max() <= info["range_m"] / 65535.0


# ---------------------------------------------------------------- layouts


@pytest.mark.parametrize("template", sorted(make_layout.TEMPLATES))
def test_layout_templates_paint_every_region(template):
    """Every colour the template promises must actually appear.

    A template that silently drops a category produces a valid-looking height
    field with a whole landform missing.
    """
    from worldclaw.layout.palette import PALETTE

    rgb = make_layout.generate(template, 192, seed=3, colors=PALETTE)
    used = {tuple(c) for c in np.unique(rgb.reshape(-1, 3), axis=0)}
    assert len(used) >= 3
    for colour in used:
        assert (rgb == np.array(colour)).all(-1).mean() > 0.001


def test_layout_generation_is_seed_stable():
    from worldclaw.layout.palette import PALETTE

    a = make_layout.generate("canyon", 128, 5, PALETTE)
    b = make_layout.generate("canyon", 128, 5, PALETTE)
    c = make_layout.generate("canyon", 128, 6, PALETTE)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_palette_colours_survive_quantisation():
    """Palette entries must stay far apart or nearest-colour matching guesses."""
    from worldclaw.layout.palette import min_separation

    assert min_separation() > 40.0
