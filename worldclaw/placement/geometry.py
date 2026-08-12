"""Ray casting and projection for object placement.

Frame conventions, stated once and used everywhere in this package
--------------------------------------------------------------------
All camera maths is in the **computer-vision convention**: camera space has x
right, y down, z forward, and a point ``P`` projects to

    u = fx * P.x / P.z + cx        v = fy * P.y / P.z + cy

with image row 0 at the top.  Blender's own camera looks down its local -z with
+y up, which is a *different* convention; the Blender stage therefore writes
``E_world_to_cam_cv`` alongside the raw matrix and this package consumes only
that one.  Mixing the two silently mirrors the scene vertically, and the failure
looks like a plausible-but-wrong placement rather than an error.

World space is the terrain frame from ``terrain/frame.py``: x east, y north,
z up, origin at the centre of the world.
"""

from __future__ import annotations

import numpy as np

from ..terrain.frame import sample_height_m


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-space points to pixels. Points behind the camera give NaN."""
    p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    out = np.stack([u, v], axis=1)
    out[z <= 0] = np.nan
    return out


def pixel_ray(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unit ray direction in camera space through a pixel."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    d = np.stack(
        [
            (uv[:, 0] - K[0, 2]) / K[0, 0],
            (uv[:, 1] - K[1, 2]) / K[1, 1],
            np.ones(len(uv)),
        ],
        axis=1,
    )
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cam_to_world(points_cam: np.ndarray, E_world_to_cam: np.ndarray) -> np.ndarray:
    """Camera space -> world space."""
    E_inv = np.linalg.inv(np.asarray(E_world_to_cam, dtype=np.float64))
    p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
    h = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    return (h @ E_inv.T)[:, :3]


def world_to_cam(points_world: np.ndarray, E_world_to_cam: np.ndarray) -> np.ndarray:
    E = np.asarray(E_world_to_cam, dtype=np.float64)
    p = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    h = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    return (h @ E.T)[:, :3]


# ---------------------------------------------------------------- terrain ray


def ray_terrain_intersection(
    origin_world: np.ndarray,
    direction_world: np.ndarray,
    height_m: np.ndarray,
    cell_size_m: float,
    t_max: float | None = None,
    coarse_steps: int = 2048,
    refine_iters: int = 40,
) -> tuple[np.ndarray, float] | tuple[None, None]:
    """First positive intersection of a world-space ray with the height field.

    March coarsely to bracket the first sign change of ``ray.z - H(ray.xy)``,
    then bisect.  Marching alone would quantise the anchor to the step length,
    and the anchor's depth divides directly into the placement scale, so the
    error would show up as objects of the wrong size rather than in the wrong
    place.
    """
    o = np.asarray(origin_world, dtype=np.float64)
    d = np.asarray(direction_world, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    h, w = height_m.shape
    if t_max is None:
        # Far enough to cross the whole tile diagonally plus its relief.
        span = np.hypot(w * cell_size_m, h * cell_size_m)
        t_max = float(span + (height_m.max() - height_m.min()) * 2.0)

    def gap(t: float) -> float:
        p = o + t * d
        return float(p[2] - sample_height_m(height_m, cell_size_m, p[0], p[1]))

    t_prev, g_prev = 0.0, gap(0.0)
    if g_prev < 0.0:
        # The origin is already underground; nothing sensible to anchor to.
        return None, None

    for i in range(1, coarse_steps + 1):
        t = t_max * i / coarse_steps
        g = gap(t)
        if g <= 0.0:
            lo, hi = t_prev, t
            for _ in range(refine_iters):
                mid = 0.5 * (lo + hi)
                if gap(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            t_hit = 0.5 * (lo + hi)
            return o + t_hit * d, t_hit
        t_prev, g_prev = t, g
    return None, None


# ------------------------------------------------------------------ mesh ray


def ray_mesh_intersection(
    origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, float] | tuple[None, None]:
    """Nearest positive ray-triangle hit (Moller-Trumbore, vectorised)."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.intp)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-12
    inv_det = np.zeros_like(det)
    inv_det[ok] = 1.0 / det[ok]

    tvec = o - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    w = np.einsum("j,ij->i", d, qvec) * inv_det
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det

    hit = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9) & (t > 1e-9)
    if not hit.any():
        return None, None
    idx = int(np.argmin(np.where(hit, t, np.inf)))
    return o + t[idx] * d, float(t[idx])


def projected_bbox(points_cam: np.ndarray, K: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned pixel bounding box of projected points."""
    uv = project(points_cam, K)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()))


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)
