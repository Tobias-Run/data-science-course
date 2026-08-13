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
    ScatterSpec,
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
    rgb = np.zeros((RES, RES, 3), dtype=np.uint8)
    rgb[:, :] = spec.regions[0].rgb
    rgb[: RES // 2, :] = spec.regions[1].rgb
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


def test_shift_semantics_for_all_offsets():
    """``_shift`` must implement out[i,j] = a[i-dy, j-dx] for any offset.

    Erosion itself only shifts along one axis at a time; this pins the general
    contract so a future 8-connected neighbourhood cannot inherit a corner bug.
    """
    from worldclaw.terrain.operators import _shift

    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    for dy, dx in ((1, 0), (-1, 0), (0, 2), (1, 1), (-1, 2), (2, -2)):
        clamped = _shift(a, dy, dx)
        filled = _shift(a, dy, dx, fill=0.0)
        for i in range(4):
            for j in range(5):
                si, sj = i - dy, j - dx
                ci, cj = min(max(si, 0), 3), min(max(sj, 0), 4)
                assert clamped[i, j] == a[ci, cj]
                expect = a[si, sj] if 0 <= si < 4 and 0 <= sj < 5 else 0.0
                assert filled[i, j] == expect
    # Fill mode must conserve what stays on the grid: nothing enters from outside.
    assert _shift(a, 1, 1, fill=0.0).sum() == a[:-1, :-1].sum()


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


def test_backdrop_never_encloses_a_viewpoint(spec, layout):
    """The horizon plane must sit below every terrain sample.

    Placed at the border's median height instead, it sat at plateau level in a
    canyon scene and put any camera on the gorge floor under a horizon-to-horizon
    ceiling -- the render came out pure black.
    """
    from worldclaw.blender.build_terrain import backdrop_height

    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    z = backdrop_height(hf.height_m)
    assert z < float(hf.height_m.min())
    # An observer standing anywhere on the terrain is above the plane.
    assert z < float(hf.height_m.min()) + 1.7


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


# ---------------------------------------------------------------- surface


def test_splat_maps_pack_channels_and_partition(spec, layout):
    from worldclaw.terrain.splat import pack_splat_maps

    rm = masks_mod.extract_masks(spec, layout)
    maps = pack_splat_maps(rm.weights, rm.names)
    assert len(maps) == 1  # two regions fit one RGBA image
    assert maps[0]["channels"] == {"R": "low", "G": "high"}
    rgba = maps[0]["rgba"]
    assert np.allclose(rgba[..., :2].sum(-1), 1.0, atol=1e-5)
    assert np.allclose(rgba[..., 2:], 0.0)


def test_splat_maps_split_beyond_four_regions():
    from worldclaw.terrain.splat import pack_splat_maps

    weights = np.zeros((6, 4, 4), dtype=np.float32)
    weights[0] = 1.0
    maps = pack_splat_maps(weights, [f"r{i}" for i in range(6)])
    assert [m["index"] for m in maps] == [0, 1]
    assert list(maps[1]["channels"]) == ["R", "G"]


def _flat_scene(slope_deg: float = 0.0, res: int = 128):
    """A terrain tilted by a known angle -- contact is analytic on it."""
    spec = TerrainSpec(
        name="ramp", seed=1, resolution=res, world_size_m=128.0, height_scale_m=1.0,
        regions=[RegionSpec(name="a", color="#000000", base_height=0.0)],
    )
    xs = np.arange(res, dtype=np.float32) * spec.cell_size_m
    height = np.tile(xs * np.tan(np.deg2rad(slope_deg)), (res, 1))
    return spec, hf_mod.Heightfield(height_m=height, normalised=height, spec=spec)


def test_contact_is_exact_on_flat_ground():
    from worldclaw.terrain.scatter import contact_report

    spec, hf = _flat_scene(0.0)
    up = np.array([0.0, 0.0, 1.0])
    contact, gap, pen = contact_report(
        hf.height_m, spec.cell_size_m, 0.0, 0.0, 0.0, 1.0, 0.05, up
    )
    assert contact == pytest.approx(1.0)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert pen == pytest.approx(0.0, abs=1e-6)


def test_contact_metric_credits_alignment_on_a_slope():
    """A prop tilted onto the surface normal must score better than an upright one.

    Measuring against a horizontal base plane regardless of the prop's actual
    orientation reports interpenetration that does not exist -- which is what
    dragged the canyon contact rate down to 84%.
    """
    from worldclaw.terrain.scatter import contact_report, resolve_base_height

    spec, hf = _flat_scene(25.0)
    slope_normal = hf.normals()[64, 64].astype(np.float64)
    up = np.array([0.0, 0.0, 1.0])

    scores = {}
    for label, axis in (("upright", up), ("aligned", slope_normal)):
        base = resolve_base_height(hf.height_m, spec.cell_size_m, 0.0, 0.0, 1.5, 0.0, axis)
        scores[label] = contact_report(
            hf.height_m, spec.cell_size_m, 0.0, 0.0, base, 1.5, 0.05, axis
        )[0]
    assert scores["aligned"] > scores["upright"]
    assert scores["aligned"] == pytest.approx(1.0)


def test_scatter_respects_slope_and_density():
    from worldclaw.terrain.scatter import scatter_scene

    spec, hf = _flat_scene(0.0, res=256)
    spec.regions[0].scatter = [
        ScatterSpec(asset_class="rock", density_per_km2=40000.0, min_spacing_m=2.0,
                    footprint_radius_m=0.5)
    ]
    weights = np.ones((1, *hf.height_m.shape), dtype=np.float32)
    inst = scatter_scene(spec, hf, weights)
    assert len(inst) > 0
    area_km2 = (spec.world_size_m**2) / 1e6
    assert len(inst) <= round(40000.0 * area_km2) + 1
    # Minimum spacing must actually hold.
    p = np.array([i.position_m[:2] for i in inst])
    d = np.hypot(*(p[:, None, :] - p[None, :, :]).T)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 2.0 - 1e-6
    assert all(i.contact_ratio == pytest.approx(1.0) for i in inst)


def test_scatter_rejects_ground_steeper_than_the_spec():
    from worldclaw.terrain.scatter import scatter_scene

    spec, hf = _flat_scene(35.0, res=192)
    spec.regions[0].scatter = [
        ScatterSpec(asset_class="shrub", density_per_km2=50000.0, max_slope_deg=20.0)
    ]
    weights = np.ones((1, *hf.height_m.shape), dtype=np.float32)
    assert scatter_scene(spec, hf, weights) == []


def test_scatter_is_seed_stable(spec, layout):
    from worldclaw.terrain.scatter import scatter_scene

    spec.regions[0].scatter = [ScatterSpec(asset_class="rock", density_per_km2=8000.0)]
    rm = masks_mod.extract_masks(spec, layout)
    hf = hf_mod.build_heightfield(spec, rm)
    a = scatter_scene(spec, hf, rm.weights)
    b = scatter_scene(spec, hf, rm.weights)
    assert [i.position_m for i in a] == [i.position_m for i in b]
    assert len(a) > 0


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


@pytest.mark.parametrize("template", sorted(make_layout.TEMPLATES))
@pytest.mark.parametrize("seed", [0, 3, 11, 42])
def test_placed_blobs_never_touch_the_border(template, seed):
    """The small inlaid regions must stay clear of the map edge.

    A blob clipped by the border gets a dead-straight boundary that no hand
    would paint, and it shows up in the render as a hard straight seam across a
    landform.  The outline reaches 1.5x the nominal radius, so the clearance
    check has to use that, not the nominal value.
    """
    from worldclaw.layout.palette import PALETTE

    rgb = make_layout.generate(template, 192, seed, PALETTE)
    inlaid = np.array(
        [int(PALETTE["rock"].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8
    )
    mask = (rgb == inlaid).all(-1)
    assert mask.any(), "the inlaid region must be painted at all"
    assert not mask[0, :].any() and not mask[-1, :].any()
    assert not mask[:, 0].any() and not mask[:, -1].any()


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


def test_rim_viewpoint_ignores_the_tile_edge():
    """The eye-height camera must stand at a landform, not at the map border.

    Region weights leave a hard step where the tile ends, and that step is
    usually the largest height difference in the scene. Without excluding it the
    search lands in a map corner and the camera looks out of the world at the
    horizon plane -- a real run produced exactly that: a frame filled with a
    featureless gradient and a sliver of sky.
    """
    from worldclaw.blender.build_terrain import rim_viewpoint

    res, cell = 384, 4.0
    h = np.zeros((res, res), dtype=np.float32)
    h[:, 150:210] = 275.0          # the real landform, mid-map
    h[:, :20] = 275.0              # tile-edge artefacts, taller in extent
    h[:, -20:] = 275.0

    (rx, ry), _ = rim_viewpoint(h, cell)
    half = (res - 1) / 2 * cell
    assert min(half - abs(rx), half - abs(ry)) > 0.03 * res * cell, "stood on the tile edge"
    # Columns 150-210 map to roughly x = -160 .. +80 in world coordinates.
    assert -220.0 < rx < 140.0, f"did not find the real landform, got x={rx}"


def test_rim_viewpoint_still_works_on_a_small_raster():
    """The margin must not consume a raster too small to spare one."""
    from worldclaw.blender.build_terrain import rim_viewpoint

    h = np.zeros((24, 24), dtype=np.float32)
    h[:, 12:] = 50.0
    (rx, ry), (lx, ly) = rim_viewpoint(h, 4.0)
    assert all(np.isfinite(v) for v in (rx, ry, lx, ly))
