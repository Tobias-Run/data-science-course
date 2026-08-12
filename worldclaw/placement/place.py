"""Ray-pair object placement (paper section 3.2).

The chain, with the reason each step exists:

1. **Crop bookkeeping.**  The object was cut out of the composition image and
   enlarged.  With ``A_i`` the affine from composition to object-centred pixel
   coordinates, the equivalent intrinsics are ``K_hat_i = A_i @ K_t`` and the
   extrinsics are unchanged -- cropping and scaling move the *image coordinate
   system*, not the camera.  This is why reconstructing from a high-resolution
   crop costs nothing in placement accuracy.

2. **Scale calibration.**  Single-image reconstruction has scale drift.  A
   factor about the mesh centre is fitted until the projected bounding-box area
   matches the reference mask's, within asymmetric tolerance: oversize is
   suppressed harder than undersize, because an object that is too big reads as
   broken while one slightly too small reads as distant.

3. **Two rays.**  One through the object's centre into the reconstructed mesh
   gives the reference point ``P_o`` at depth ``Z_o``.  The crop centre mapped
   back through ``A_i^-1`` gives a pixel in the composition image; a ray from
   the terrain camera through it, intersected with the terrain, gives the anchor
   ``P_t`` at depth ``Z_t``.

4. **Similarity transform.**  ``s = (Z_t / Z_o) * (f_o / f_hat_i)`` makes the
   object subtend the same angle from the terrain camera that it did in its own
   reconstruction, and the translation puts ``P_o`` exactly on ``P_t``.

5. **Contact search.**  Anchor depth and scale are varied *together* along the
   terrain ray, which leaves the 2D projection untouched -- the object keeps
   looking exactly as composed while it settles onto the ground.

The double-transform trap
-------------------------
``T_place`` ends in ``@ T_l2c``.  If the exporter already baked the local-to-
camera transform into the vertices, applying it again silently doubles the
rotation and translation.  ``PlacementRecord.l2c_baked_into_vertices`` records
which case a given asset is, and ``compose_transform`` refuses to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schemas import PlacementRecord
from .geometry import (
    bbox_area,
    cam_to_world,
    pixel_ray,
    project,
    projected_bbox,
    ray_mesh_intersection,
    ray_terrain_intersection,
)


def crop_affine(x0: float, y0: float, scale: float) -> np.ndarray:
    """Affine ``A_i``: composition pixels -> object-centred crop pixels.

    Crop at ``(x0, y0)`` then enlarge by ``scale``.
    """
    return np.array(
        [[scale, 0.0, -scale * x0], [0.0, scale, -scale * y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def equivalent_intrinsics(A: np.ndarray, K_t: np.ndarray) -> np.ndarray:
    """``K_hat_i = A_i @ K_t`` -- the intrinsics the crop behaves as if it had."""
    return np.asarray(A, dtype=np.float64) @ np.asarray(K_t, dtype=np.float64)


def crop_to_composition(A: np.ndarray, uv_crop: np.ndarray) -> np.ndarray:
    """Map a crop pixel back to the composition image through ``A_i^-1``."""
    uv = np.atleast_2d(np.asarray(uv_crop, dtype=np.float64))
    h = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    back = (np.linalg.inv(np.asarray(A, dtype=np.float64)) @ h.T).T
    return back[:, :2] / back[:, 2:3]


# --------------------------------------------------------------- calibration


@dataclass
class CalibrationResult:
    scale: float
    area_ratio: float
    converged: bool
    iterations: int


def calibrate_scale(
    vertices_cam: np.ndarray,
    K_hat: np.ndarray,
    reference_area_px: float,
    eps_under: float = 0.10,
    eps_over: float = 0.05,
    max_iters: int = 40,
) -> CalibrationResult:
    """Fit ``lambda_i`` about the mesh centre so the projected area matches.

    Asymmetric tolerance ``[-eps_under, +eps_over]``: the accepted band is wider
    below the target than above it, so the search settles on the small side of
    ambiguity.

    Scaling about the centroid moves the mesh in depth as well as size, so the
    projected area is not exactly quadratic in the factor -- hence a bisection
    on the log factor rather than a closed form.
    """
    v = np.asarray(vertices_cam, dtype=np.float64)
    centre = v.mean(axis=0)

    def area_at(lam: float) -> float:
        return bbox_area(projected_bbox(centre + (v - centre) * lam, K_hat))

    def ratio_at(lam: float) -> float:
        a = area_at(lam)
        return a / reference_area_px if reference_area_px > 0 else np.inf

    r1 = ratio_at(1.0)
    if -eps_under <= r1 - 1.0 <= eps_over:
        return CalibrationResult(1.0, r1, True, 0)

    lo, hi = 1.0, 1.0
    for _ in range(24):  # bracket first; the relation is monotonic in lambda
        if ratio_at(hi) < 1.0:
            hi *= 1.5
        else:
            break
    for _ in range(24):
        if ratio_at(lo) > 1.0:
            lo /= 1.5
        else:
            break

    lam = 1.0
    for i in range(1, max_iters + 1):
        lam = float(np.sqrt(lo * hi))  # geometric midpoint
        r = ratio_at(lam)
        if -eps_under <= r - 1.0 <= eps_over:
            return CalibrationResult(lam, r, True, i)
        if r < 1.0:
            lo = lam
        else:
            hi = lam
    return CalibrationResult(lam, ratio_at(lam), False, max_iters)


# ----------------------------------------------------------------- placement


@dataclass
class PlacementInputs:
    """Everything the ray pair needs, named after the paper's symbols."""

    # terrain camera
    K_t: np.ndarray
    E_t: np.ndarray  # world -> terrain camera, CV convention
    height_m: np.ndarray
    cell_size_m: float
    # crop
    A_i: np.ndarray
    crop_size_px: tuple[int, int]
    # reconstruction
    vertices_obj_cam: np.ndarray  # mesh in the object camera's frame
    faces: np.ndarray
    f_o: float  # object camera focal length, pixels
    R_o: np.ndarray  # world -> object camera rotation
    reference_area_px: float = 0.0
    l2c_baked_into_vertices: bool = True
    T_l2c: np.ndarray | None = None


@dataclass
class PlacementResult:
    scale: float
    rotation: np.ndarray
    translation_cam: np.ndarray
    anchor_world: np.ndarray
    P_o: np.ndarray
    Z_o: float
    Z_t: float
    calibration: CalibrationResult
    transform_world: np.ndarray = field(default_factory=lambda: np.eye(4))
    contact_ratio: float = 0.0
    contact_passed: bool = False
    notes: list[str] = field(default_factory=list)


def compose_transform(
    s: float, R: np.ndarray, t: np.ndarray, T_l2c: np.ndarray | None, baked: bool
) -> np.ndarray:
    """Assemble ``T_place``, applying ``T_l2c`` only when it is not already baked.

    Refuses to guess: asking for the unbaked case without supplying ``T_l2c`` is
    an error, because silently skipping it produces a transform that looks
    reasonable and puts every object in the wrong pose.
    """
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    if baked:
        return T
    if T_l2c is None:
        raise ValueError(
            "l2c_baked_into_vertices is False but no T_l2c was supplied; "
            "the local-to-camera transform must be applied exactly once"
        )
    return T @ np.asarray(T_l2c, dtype=np.float64)


def place_object(inp: PlacementInputs) -> PlacementResult:
    """Run the full ray pair and return the placement transform."""
    notes: list[str] = []
    K_hat = equivalent_intrinsics(inp.A_i, inp.K_t)
    f_hat = float(K_hat[0, 0])

    # -- 2. scale calibration in image space ------------------------------
    if inp.reference_area_px > 0:
        calib = calibrate_scale(inp.vertices_obj_cam, K_hat, inp.reference_area_px)
        if not calib.converged:
            notes.append(f"scale calibration did not converge (ratio {calib.area_ratio:.3f})")
    else:
        calib = CalibrationResult(1.0, 1.0, True, 0)
    v = np.asarray(inp.vertices_obj_cam, dtype=np.float64)
    centre = v.mean(axis=0)
    v_cal = centre + (v - centre) * calib.scale

    # -- 3a. first ray: object centre into the reconstructed mesh ---------
    # Through the *object camera's* intrinsics, not the crop's.  The mesh lives
    # in the object camera's frame, and its image is the crop, so that camera's
    # principal point is the crop centre and its focal is f_o.  Using K_hat here
    # shoots a ray expressed in the terrain camera's frame into a mesh that is
    # not in that frame -- plausible-looking and wrong.
    cw, ch = inp.crop_size_px
    centre_px = np.array([[cw / 2.0, ch / 2.0]])
    K_o = np.array([[inp.f_o, 0.0, cw / 2.0], [0.0, inp.f_o, ch / 2.0], [0.0, 0.0, 1.0]])
    d_obj = pixel_ray(centre_px, K_o)[0]
    P_o, _ = ray_mesh_intersection(np.zeros(3), d_obj, v_cal, inp.faces)
    if P_o is None:
        # The crop centre may miss a concave silhouette; the centroid is the
        # honest fallback and is reported rather than hidden.
        P_o = v_cal.mean(axis=0)
        notes.append("centre ray missed the mesh; used its centroid as P_o")
    Z_o = float(P_o[2])
    if Z_o <= 0:
        raise ValueError("reference point is behind the object camera")

    # -- 3b. second ray: crop centre -> composition pixel -> terrain ------
    uv_comp = crop_to_composition(inp.A_i, centre_px)
    d_cam = pixel_ray(uv_comp, inp.K_t)[0]
    cam_origin_world = cam_to_world(np.zeros((1, 3)), inp.E_t)[0]
    d_world = cam_to_world(d_cam[None, :], inp.E_t)[0] - cam_origin_world
    P_t_world, _ = ray_terrain_intersection(
        cam_origin_world, d_world, inp.height_m, inp.cell_size_m
    )
    if P_t_world is None:
        raise ValueError("terrain ray missed the height field: nothing to anchor to")
    P_t_cam = np.asarray(
        (np.concatenate([P_t_world, [1.0]]) @ np.asarray(inp.E_t).T)[:3], dtype=np.float64
    )
    Z_t = float(P_t_cam[2])

    # -- 4. similarity transform -----------------------------------------
    R_t = np.asarray(inp.E_t, dtype=np.float64)[:3, :3]
    R_i = R_t @ np.asarray(inp.R_o, dtype=np.float64).T
    s = (Z_t / Z_o) * (inp.f_o / f_hat)
    t = P_t_cam - s * (R_i @ P_o)

    T_place = compose_transform(s, R_i, t, inp.T_l2c, inp.l2c_baked_into_vertices)
    T_world = np.linalg.inv(np.asarray(inp.E_t, dtype=np.float64)) @ T_place

    return PlacementResult(
        scale=s,
        rotation=R_i,
        translation_cam=t,
        anchor_world=P_t_world,
        P_o=P_o,
        Z_o=Z_o,
        Z_t=Z_t,
        calibration=calib,
        transform_world=T_world,
        notes=notes,
    )


# ------------------------------------------------------------ contact search


def contact_ratio_for_mesh(
    vertices_world: np.ndarray,
    height_m: np.ndarray,
    cell_size_m: float,
    tolerance_m: float,
    bottom_fraction: float = 0.15,
) -> tuple[float, float, float]:
    """Contact of an object's lowest vertices with the terrain.

    The same measurement M2 applies to scattered props, on a reconstructed mesh:
    share of the bottom band within tolerance, plus the largest float and the
    largest penetration.  Keeping one definition means the contact rate is
    comparable across milestones.
    """
    from ..terrain.frame import sample_height_m

    v = np.asarray(vertices_world, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    z = v[:, 2]
    cutoff = z.min() + bottom_fraction * max(z.max() - z.min(), 1e-9)
    bottom = v[z <= cutoff]
    if len(bottom) == 0:
        bottom = v[np.argsort(z)[: max(1, len(v) // 20)]]

    ground = np.asarray(
        sample_height_m(height_m, cell_size_m, bottom[:, 0], bottom[:, 1]), dtype=np.float64
    )
    delta = bottom[:, 2] - ground
    contact = float((np.abs(delta) <= tolerance_m).mean())
    return contact, float(max(delta.max(), 0.0)), float(max(-delta.min(), 0.0))


def contact_search(
    result: PlacementResult,
    inp: PlacementInputs,
    vertices_obj_cam: np.ndarray,
    tolerance_m: float = 0.25,
    threshold: float = 0.35,
    depth_range: tuple[float, float] = (0.75, 1.35),
    samples: int = 41,
) -> PlacementResult:
    """Slide the anchor along its ray, scaling in step, until the object lands.

    Moving the anchor to depth ``k * Z_t`` and scaling by the same ``k`` leaves
    the projection identical, so the object keeps looking exactly as composed
    while its contact with the ground changes.  That invariance is what makes
    this search free of visual cost -- and it is the property to protect if the
    search is ever extended.
    """
    E_inv = np.linalg.inv(np.asarray(inp.E_t, dtype=np.float64))
    R_t = np.asarray(inp.E_t, dtype=np.float64)[:3, :3]
    R_i = R_t @ np.asarray(inp.R_o, dtype=np.float64).T
    d_cam = result.anchor_world  # only used for reporting
    del d_cam

    P_t_cam_unit = np.asarray(
        (np.concatenate([result.anchor_world, [1.0]]) @ np.asarray(inp.E_t).T)[:3]
    ) / result.Z_t

    best = None
    for k in np.linspace(depth_range[0], depth_range[1], samples):
        Z = result.Z_t * float(k)
        P_t_cam = P_t_cam_unit * Z
        s = (Z / result.Z_o) * (inp.f_o / float(equivalent_intrinsics(inp.A_i, inp.K_t)[0, 0]))
        t = P_t_cam - s * (R_i @ result.P_o)
        T = compose_transform(s, R_i, t, inp.T_l2c, inp.l2c_baked_into_vertices)
        T_world = E_inv @ T

        v = np.asarray(vertices_obj_cam, dtype=np.float64)
        h = np.concatenate([v, np.ones((len(v), 1))], axis=1)
        v_world = (h @ T_world.T)[:, :3]
        contact, gap, pen = contact_ratio_for_mesh(
            v_world, inp.height_m, inp.cell_size_m, tolerance_m
        )
        score = (contact, -pen)
        if best is None or score > best[0]:
            best = (score, s, t, T_world, contact, gap, pen, P_t_cam)
        if contact >= threshold and pen <= tolerance_m:
            break

    assert best is not None
    _, s, t, T_world, contact, gap, pen, P_t_cam = best
    result.scale = s
    result.translation_cam = t
    result.transform_world = T_world
    result.anchor_world = cam_to_world(P_t_cam[None, :], inp.E_t)[0]
    result.contact_ratio = contact
    result.contact_passed = contact >= threshold and pen <= tolerance_m
    if not result.contact_passed:
        result.notes.append(
            f"contact search ended below threshold: {contact:.2f} "
            f"(gap {gap:.2f} m, penetration {pen:.2f} m)"
        )
    return result


def to_record(
    result: PlacementResult, object_id: str, category: str, region: str, mesh_path: str,
    inp: PlacementInputs, refine_iterations: int = 0,
) -> PlacementRecord:
    return PlacementRecord(
        object_id=object_id,
        category=category,
        region=region,
        mesh_path=mesh_path,
        anchor_world_m=tuple(float(x) for x in result.anchor_world),
        reference_point_cam=tuple(float(x) for x in result.P_o),
        depth_anchor_m=result.Z_t,
        depth_object_m=result.Z_o,
        scale=result.scale,
        rotation_matrix=[[float(x) for x in row] for row in result.rotation],
        translation_m=tuple(float(x) for x in result.translation_cam),
        l2c_baked_into_vertices=inp.l2c_baked_into_vertices,
        calibration_scale=result.calibration.scale,
        contact_ratio=result.contact_ratio,
        contact_passed=result.contact_passed,
        refine_iterations=refine_iterations,
    )
