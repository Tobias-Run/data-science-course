"""Tests for the ray-pair placement (paper section 3.2).

The load-bearing test is the round trip: take a known object at a known pose,
compute what the cameras would have seen, run the placement, and check it
recovers the pose it started from.  Nothing else proves the frame conventions,
the crop bookkeeping and the scale formula agree with each other -- and all
three are easy to get individually plausible and jointly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldclaw.placement.geometry import (
    cam_to_world,
    pixel_ray,
    project,
    ray_mesh_intersection,
    ray_terrain_intersection,
    world_to_cam,
)
from worldclaw.placement.place import (
    PlacementInputs,
    calibrate_scale,
    compose_transform,
    contact_ratio_for_mesh,
    contact_search,
    crop_affine,
    crop_to_composition,
    equivalent_intrinsics,
    place_object,
)


def make_K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def look_at_extrinsics(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World -> camera (CV convention: x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # rows: camera axes in world coords
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ eye
    return E


def unit_box(size: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A box with its base on z = 0 in its own local frame."""
    s = size / 2.0
    v = np.array([
        [-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0],
        [-s, -s, size], [s, -s, size], [s, s, size], [-s, s, size],
    ], dtype=np.float64)
    f = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.intp)
    return v, f


def flat_terrain(res: int = 128, cell: float = 2.0, height: float = 0.0) -> np.ndarray:
    return np.full((res, res), height, dtype=np.float32)


def sloped_terrain(res: int = 192, cell: float = 2.0, slope_deg: float = 12.0) -> np.ndarray:
    xs = (np.arange(res, dtype=np.float64) - (res - 1) / 2.0) * cell
    return np.tile(xs * np.tan(np.deg2rad(slope_deg)), (res, 1)).astype(np.float32)


# ------------------------------------------------------------------ geometry


def test_projection_and_ray_are_inverse():
    K = make_K(600.0, 640, 480)
    pts = np.array([[0.4, -0.2, 5.0], [-1.5, 0.8, 12.0], [0.0, 0.0, 3.0]])
    uv = project(pts, K)
    rays = pixel_ray(uv, K)
    for p, r in zip(pts, rays):
        assert np.allclose(r, p / np.linalg.norm(p), atol=1e-9)


def test_points_behind_the_camera_do_not_project():
    K = make_K(500.0, 320, 240)
    assert np.isnan(project(np.array([[0.0, 0.0, -2.0]]), K)).all()


def test_world_and_camera_round_trip():
    E = look_at_extrinsics(np.array([100.0, -80.0, 60.0]), np.zeros(3))
    pts = np.array([[10.0, 20.0, 5.0], [-30.0, 0.0, 12.0]])
    assert np.allclose(cam_to_world(world_to_cam(pts, E), E), pts, atol=1e-9)


def test_ray_hits_flat_terrain_at_the_expected_point():
    terrain = flat_terrain(res=128, height=10.0)
    origin = np.array([0.0, 0.0, 110.0])
    hit, t = ray_terrain_intersection(origin, np.array([0.0, 0.0, -1.0]), terrain, 2.0)
    assert hit is not None
    assert hit[2] == pytest.approx(10.0, abs=1e-3)
    assert t == pytest.approx(100.0, abs=1e-3)


def test_ray_hits_a_slope_where_geometry_says_it_should():
    cell, slope = 2.0, 20.0
    terrain = sloped_terrain(res=256, cell=cell, slope_deg=slope)
    # Straight down from a known xy: the hit must be the terrain height there.
    x = 40.0
    hit, _ = ray_terrain_intersection(
        np.array([x, 0.0, 400.0]), np.array([0.0, 0.0, -1.0]), terrain, cell
    )
    assert hit is not None
    assert hit[2] == pytest.approx(x * np.tan(np.deg2rad(slope)), abs=0.05)


def test_ray_returns_nothing_when_it_leaves_the_tile():
    terrain = flat_terrain(res=64, height=0.0)
    hit, _ = ray_terrain_intersection(
        np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, 1.0]), terrain, 2.0
    )
    assert hit is None


def test_ray_mesh_intersection_takes_the_nearest_face():
    v, f = unit_box(2.0)
    hit, t = ray_mesh_intersection(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]), v, f)
    assert hit is not None
    assert hit[2] == pytest.approx(2.0, abs=1e-9)  # the top face, not the base
    assert t == pytest.approx(8.0, abs=1e-9)


# ---------------------------------------------------------------------- crop


def test_equivalent_intrinsics_match_an_actually_cropped_projection():
    """K_hat = A @ K must equal projecting into the cropped image directly.

    This is the identity the paper leans on to claim resolution costs nothing:
    if it does not hold exactly, every crop introduces a placement bias.
    """
    K_t = make_K(800.0, 1024, 768)
    A = crop_affine(x0=300.0, y0=200.0, scale=3.0)
    K_hat = equivalent_intrinsics(A, K_t)

    pts = np.array([[0.5, 0.3, 8.0], [-1.2, 0.9, 15.0], [2.0, -1.0, 25.0]])
    uv_full = project(pts, K_t)
    expected = (uv_full - np.array([300.0, 200.0])) * 3.0
    assert np.allclose(project(pts, K_hat), expected, atol=1e-9)


def test_crop_centre_maps_back_to_the_composition_pixel():
    A = crop_affine(x0=120.0, y0=64.0, scale=4.0)
    crop_centre = np.array([[256.0, 256.0]])
    back = crop_to_composition(A, crop_centre)
    assert np.allclose(back, [[120.0 + 64.0, 64.0 + 64.0]], atol=1e-9)


# --------------------------------------------------------------- calibration


def test_calibration_recovers_a_known_scale_error():
    K = make_K(700.0, 512, 512)
    v, _ = unit_box(2.0)
    v_cam = v + np.array([0.0, 0.0, 12.0])
    from worldclaw.placement.geometry import bbox_area, projected_bbox

    target = bbox_area(projected_bbox(v_cam, K))
    # Pretend the reconstruction came out 1.6x too large.
    centre = v_cam.mean(axis=0)
    drifted = centre + (v_cam - centre) * 1.6

    result = calibrate_scale(drifted, K, target)
    assert result.converged
    assert result.scale == pytest.approx(1.0 / 1.6, rel=0.08)


def test_calibration_tolerance_is_asymmetric():
    """Oversize must be suppressed harder than undersize."""
    K = make_K(700.0, 512, 512)
    v, _ = unit_box(2.0)
    v_cam = v + np.array([0.0, 0.0, 12.0])
    from worldclaw.placement.geometry import bbox_area, projected_bbox

    area = bbox_area(projected_bbox(v_cam, K))
    under = calibrate_scale(v_cam, K, area / 0.93, eps_under=0.10, eps_over=0.05)
    over = calibrate_scale(v_cam, K, area / 1.07, eps_under=0.10, eps_over=0.05)
    # 7% undersize is inside the band and accepted as-is; 7% oversize is not.
    assert under.iterations == 0
    assert over.iterations > 0


# ------------------------------------------------------------- the round trip


def _round_trip_case(
    slope_deg: float, crop_scale: float, f_o: float, obj_size: float,
    recon_distance: float = 25.0,
):
    """Build a scene where the true placement is known, then recover it.

    The subtlety that makes or breaks this fixture: a single-image
    reconstruction is **not** metrically true.  It comes back at whatever scale
    reproduces the crop's appearance from its own camera, which is precisely why
    the paper needs ``s = (Z_t/Z_o)(f_o/f_hat)`` at all.  Building the fixture
    with a metrically correct mesh instead makes the placement look 3.5x wrong
    when it is in fact doing exactly its job.

    So the synthetic reconstruction is scaled by ``lambda`` such that its
    apparent size from the object camera equals the object's apparent size in
    the crop.  The placement then has to undo exactly that.
    """
    cell = 2.0
    terrain = sloped_terrain(res=256, cell=cell, slope_deg=slope_deg)
    f_t = 900.0
    K_t = make_K(f_t, 1024, 768)
    eye = np.array([-160.0, -220.0, 140.0])
    E_t = look_at_extrinsics(eye, np.array([0.0, 0.0, 0.0]))

    # The truth: an object standing on the terrain at this world position.
    from worldclaw.terrain.frame import sample_height_m

    gx, gy = 12.0, -8.0
    gz = float(sample_height_m(terrain, cell, gx, gy))
    ground_point = np.array([gx, gy, gz])

    # Where the composition image shows that ground point.
    uv_comp = project(world_to_cam(ground_point[None, :], E_t), K_t)[0]
    Z_t = float(world_to_cam(ground_point[None, :], E_t)[0][2])

    # The crop the agent would take around it.
    cw = ch = 512
    x0 = uv_comp[0] - (cw / 2.0) / crop_scale
    y0 = uv_comp[1] - (ch / 2.0) / crop_scale
    A = crop_affine(x0, y0, crop_scale)
    f_hat = crop_scale * f_t

    # The reconstruction, at the scale that reproduces the crop's appearance.
    lam = (recon_distance / Z_t) * (f_hat / f_o)
    v_local, faces = unit_box(obj_size)
    v_recon = v_local * lam

    # The reconstruction camera must view the object from the *same direction*
    # the terrain camera does -- that is what SAM3D sees, since it reconstructs
    # from this very crop. Pointing it elsewhere gives the object a different
    # world orientation, and the placement then faithfully reproduces that
    # different orientation.
    view_dir = ground_point - eye
    view_dir /= np.linalg.norm(view_dir)
    centre_local = np.array([0.0, 0.0, obj_size * lam * 0.5])
    E_o = look_at_extrinsics(centre_local - view_dir * recon_distance, centre_local)
    v_obj_cam = world_to_cam(v_recon, E_o)

    inp = PlacementInputs(
        K_t=K_t, E_t=E_t, height_m=terrain, cell_size_m=cell,
        A_i=A, crop_size_px=(cw, ch),
        vertices_obj_cam=v_obj_cam, faces=faces,
        f_o=f_o, R_o=E_o[:3, :3],
        l2c_baked_into_vertices=True,
    )
    return inp, v_obj_cam, ground_point, terrain, cell


def test_round_trip_recovers_the_anchor_on_flat_ground():
    inp, v_obj_cam, truth, terrain, cell = _round_trip_case(0.0, 3.0, 900.0, 2.0)
    res = place_object(inp)
    assert np.allclose(res.anchor_world[:2], truth[:2], atol=0.5)
    assert res.anchor_world[2] == pytest.approx(truth[2], abs=0.2)


def test_round_trip_recovers_the_anchor_on_a_slope():
    """A slope is where an anchor error turns into a visible mistake."""
    inp, v_obj_cam, truth, terrain, cell = _round_trip_case(18.0, 4.0, 750.0, 3.0)
    res = place_object(inp)
    assert np.allclose(res.anchor_world[:2], truth[:2], atol=0.6)
    assert res.anchor_world[2] == pytest.approx(truth[2], abs=0.3)


def test_round_trip_recovers_the_metric_object_size():
    """A consistently reconstructed object must be placed at its true size.

    This is what ``s = (Z_t/Z_o)(f_o/f_hat)`` is for: get it wrong and objects
    land in the right spot at the wrong size, which is the failure the paper's
    scale calibration exists to catch downstream.
    """
    obj_size = 3.0
    inp, v_obj_cam, truth, terrain, cell = _round_trip_case(0.0, 3.0, 900.0, obj_size)
    res = place_object(inp)

    h = np.concatenate([v_obj_cam, np.ones((len(v_obj_cam), 1))], axis=1)
    v_world = (h @ res.transform_world.T)[:, :3]
    extent = v_world.max(axis=0) - v_world.min(axis=0)
    assert extent[2] == pytest.approx(obj_size, rel=0.12)
    assert extent[0] == pytest.approx(obj_size, rel=0.12)


def test_placed_object_keeps_the_apparent_size_it_was_reconstructed_at():
    """The contract that holds regardless of metric scale.

    The composition image is the ground truth for appearance: whatever the
    reconstruction's own scale convention, the placed object must subtend the
    same angle in the crop that it did in its own camera.  Checked in pixels,
    so it is independent of every metric assumption in the fixture.
    """
    from worldclaw.placement.geometry import bbox_area, projected_bbox

    for crop_scale, f_o in ((3.0, 900.0), (6.0, 640.0)):
        inp, v_obj_cam, _, _, _ = _round_trip_case(10.0, crop_scale, f_o, 2.5)
        res = place_object(inp)
        K_hat = equivalent_intrinsics(inp.A_i, inp.K_t)
        K_o = make_K(f_o, *inp.crop_size_px)

        placed_cam = (v_obj_cam @ (res.scale * res.rotation).T) + res.translation_cam
        area_recon = bbox_area(projected_bbox(v_obj_cam, K_o))
        area_placed = bbox_area(projected_bbox(placed_cam, K_hat))
        assert area_placed / area_recon == pytest.approx(1.0, rel=0.15)


def test_round_trip_is_independent_of_crop_resolution():
    """Cropping and enlarging must not move the object.

    The paper's claim that reconstruction resolution costs no placement accuracy
    rests entirely on K_hat = A @ K, so it deserves a direct test.
    """
    anchors = []
    for crop_scale in (2.0, 4.0, 8.0):
        inp, _, truth, _, _ = _round_trip_case(14.0, crop_scale, 800.0, 2.5)
        anchors.append(place_object(inp).anchor_world)
    spread = np.ptp(np.array(anchors), axis=0)
    assert float(np.linalg.norm(spread)) < 0.35


def test_scale_tracks_the_object_camera_focal_length():
    """s must be proportional to f_o with everything else held fixed.

    Varying f_o through the fixture would also change the reconstruction's
    scale, so the same inputs are reused and only f_o is swapped -- otherwise
    the test measures the fixture rather than the formula.
    """
    import dataclasses

    inp, _, _, _, _ = _round_trip_case(0.0, 3.0, 600.0, 2.0)
    s600 = place_object(inp).scale
    s900 = place_object(dataclasses.replace(inp, f_o=900.0)).scale
    assert s900 / s600 == pytest.approx(900.0 / 600.0, rel=1e-9)


def test_scale_is_inversely_proportional_to_the_crop_enlargement():
    """f_hat = crop_scale * f_t, so doubling the crop halves s.

    The two effects must cancel exactly, which is what keeps the placement
    independent of the resolution the reconstruction ran at.
    """
    import dataclasses

    inp, _, _, _, _ = _round_trip_case(0.0, 3.0, 800.0, 2.0)
    cw, ch = inp.crop_size_px
    # Re-centre the enlarged crop on the same composition pixel, or the anchor
    # ray moves and the comparison measures two different placements.
    uv_comp = crop_to_composition(inp.A_i, np.array([[cw / 2.0, ch / 2.0]]))[0]
    A2 = crop_affine(uv_comp[0] - (cw / 2.0) / 6.0, uv_comp[1] - (ch / 2.0) / 6.0, 6.0)

    s1 = place_object(inp).scale
    s2 = place_object(dataclasses.replace(inp, A_i=A2)).scale
    assert s2 / s1 == pytest.approx(3.0 / 6.0, rel=1e-9)


# ------------------------------------------------------- the double transform


def test_compose_refuses_to_guess_about_the_baked_transform():
    R, t = np.eye(3), np.zeros(3)
    with pytest.raises(ValueError, match="exactly once"):
        compose_transform(1.0, R, t, None, baked=False)


def test_applying_l2c_twice_is_detectably_different():
    """Guard for the trap the briefing calls out by name."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T_l2c = np.eye(4)
    T_l2c[:3, 3] = [5.0, 0.0, 0.0]

    baked = compose_transform(1.0, R, t, T_l2c, baked=True)
    unbaked = compose_transform(1.0, R, t, T_l2c, baked=False)
    assert not np.allclose(baked, unbaked)
    assert unbaked[0, 3] == pytest.approx(6.0)  # 1 + 5, applied exactly once


# ------------------------------------------------------------ contact search


def test_contact_metric_is_perfect_for_a_box_sitting_on_flat_ground():
    terrain = flat_terrain(res=128, height=25.0)
    v, _ = unit_box(2.0)
    v_world = v + np.array([0.0, 0.0, 25.0])
    contact, gap, pen = contact_ratio_for_mesh(v_world, terrain, 2.0, tolerance_m=0.05)
    assert contact == pytest.approx(1.0)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert pen == pytest.approx(0.0, abs=1e-6)


def test_contact_metric_reports_a_floating_object():
    terrain = flat_terrain(res=128, height=0.0)
    v, _ = unit_box(2.0)
    contact, gap, pen = contact_ratio_for_mesh(v + np.array([0, 0, 3.0]), terrain, 2.0, 0.1)
    assert contact == 0.0
    assert gap == pytest.approx(3.0, abs=1e-6)
    assert pen == pytest.approx(0.0, abs=1e-6)


def test_contact_search_preserves_the_projection():
    """Sliding along the ray must not change how the object looks.

    That invariance is the whole reason the search is allowed to run: it buys
    ground contact at no visual cost. If a future change breaks it, the objects
    start drifting in the frame and this test is the alarm.
    """
    inp, v_obj_cam, _, _, _ = _round_trip_case(16.0, 3.0, 800.0, 2.5)
    res = place_object(inp)

    def silhouette(transform):
        h = np.concatenate([v_obj_cam, np.ones((len(v_obj_cam), 1))], axis=1)
        world = (h @ transform.T)[:, :3]
        return project(world_to_cam(world, inp.E_t), inp.K_t)

    before = silhouette(res.transform_world)
    after = silhouette(contact_search(res, inp, v_obj_cam).transform_world)
    assert np.allclose(before, after, atol=0.75)


def test_contact_search_improves_ground_contact():
    inp, v_obj_cam, _, terrain, cell = _round_trip_case(20.0, 3.0, 800.0, 3.0)
    res = place_object(inp)

    def contact_of(transform):
        h = np.concatenate([v_obj_cam, np.ones((len(v_obj_cam), 1))], axis=1)
        return contact_ratio_for_mesh((h @ transform.T)[:, :3], terrain, cell, 0.3)[0]

    before = contact_of(res.transform_world)
    after = contact_of(contact_search(res, inp, v_obj_cam, tolerance_m=0.3).transform_world)
    assert after >= before
