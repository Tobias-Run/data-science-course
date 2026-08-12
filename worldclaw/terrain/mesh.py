"""Height field -> triangle mesh, plus the raster exports downstream stages want.

The world frame itself lives in ``frame.py``; everything here converts through
it rather than reimplementing the convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .frame import world_from_raster
from .heightfield import Heightfield


def grid_vertices(hf: Heightfield) -> np.ndarray:
    """(H*W, 3) float32 vertex positions in world metres."""
    h, w = hf.height_m.shape
    xs, _ = world_from_raster(hf.height_m.shape, hf.spec.cell_size_m, 0, np.arange(w))
    _, ys = world_from_raster(hf.height_m.shape, hf.spec.cell_size_m, np.arange(h), 0)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), hf.height_m.ravel()], axis=1).astype(np.float32)


def grid_faces(h: int, w: int) -> np.ndarray:
    """(2*(H-1)*(W-1), 3) int32 triangle indices, counter-clockwise seen from +z."""
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    tl, tr = idx[:-1, :-1], idx[:-1, 1:]
    bl, br = idx[1:, :-1], idx[1:, 1:]
    # Row 0 is north, so "bottom" rows are further south: bl -> br -> tr is CCW
    # when viewed from above.
    t1 = np.stack([bl.ravel(), br.ravel(), tr.ravel()], axis=1)
    t2 = np.stack([bl.ravel(), tr.ravel(), tl.ravel()], axis=1)
    return np.concatenate([t1, t2], axis=0)


def grid_uvs(h: int, w: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)
    gu, gv = np.meshgrid(u, v)
    return np.stack([gu.ravel(), gv.ravel()], axis=1)


def write_obj(hf: Heightfield, path: str | Path, stride: int = 1) -> Path:
    """Write the terrain as a Wavefront OBJ.

    ``stride`` decimates the grid for a quick look; the full-resolution field
    always stays in the .npz, so this is a preview knob, not a data loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height = hf.height_m[::stride, ::stride]
    h, w = height.shape
    cell = hf.spec.cell_size_m * stride
    xs, _ = world_from_raster((h, w), cell, 0, np.arange(w))
    _, ys = world_from_raster((h, w), cell, np.arange(h), 0)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx.ravel(), gy.ravel(), height.ravel()], axis=1)
    uvs = grid_uvs(h, w)
    faces = grid_faces(h, w) + 1  # OBJ indices are 1-based

    with open(path, "w") as fh:
        fh.write(f"# worldclaw terrain '{hf.spec.name}'  {w}x{h}  "
                 f"{hf.spec.world_size_m:.1f}m\n")
        np.savetxt(fh, verts, fmt="v %.4f %.4f %.4f")
        np.savetxt(fh, uvs, fmt="vt %.6f %.6f")
        fh.write("o terrain\n")
        # vertex and uv indices coincide on a grid: "f v/vt v/vt v/vt"
        np.savetxt(fh, np.repeat(faces, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")
    return path


def write_heightmap_png16(hf: Heightfield, path: str | Path) -> tuple[Path, dict[str, float]]:
    """16-bit greyscale height map.

    This is the Unreal landscape import path (M5) and the lossless raster
    interchange in general -- 8 bit would quantise a 200 m range to 0.8 m steps,
    visible as terracing on gentle ground.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lo = float(hf.height_m.min())
    hi = float(hf.height_m.max())
    rng = max(hi - lo, 1e-6)
    q = np.clip((hf.height_m - lo) / rng, 0.0, 1.0)
    # Pillow infers I;16 from a uint16 array; passing mode= explicitly is
    # deprecated and slated for removal.
    Image.fromarray((q * 65535.0 + 0.5).astype(np.uint16)).save(path)
    return path, {"height_min_m": lo, "height_max_m": hi, "range_m": rng}


def write_preview_png(hf: Heightfield, path: str | Path, light_deg: float = 315.0) -> Path:
    """Hillshaded preview -- the fastest honest look at a height field.

    Shading uses the true surface normals, so it shows the geometry a mesh would
    have, not a colour ramp that flatters it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = hf.normals()
    az, alt = np.deg2rad(light_deg), np.deg2rad(45.0)
    light = np.array(
        [np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], dtype=np.float32
    )
    shade = np.clip((n * light).sum(-1), 0.0, 1.0)

    lo, hi = float(hf.height_m.min()), float(hf.height_m.max())
    t = (hf.height_m - lo) / max(hi - lo, 1e-6)
    ramp = np.stack([0.35 + 0.6 * t, 0.42 + 0.45 * t, 0.38 + 0.5 * t], axis=-1)
    rgb = np.clip(ramp * (0.25 + 0.85 * shade[..., None]), 0.0, 1.0)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


def save_npz(hf: Heightfield, path: str | Path, weights: np.ndarray | None = None) -> Path:
    """Persist the field itself -- the artefact every later stage reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "height_m": hf.height_m,
        "normalised": hf.normalised,
        "world_size_m": np.float32(hf.spec.world_size_m),
        "cell_size_m": np.float32(hf.spec.cell_size_m),
        "height_scale_m": np.float32(hf.spec.height_scale_m),
    }
    if weights is not None:
        payload["weights"] = weights
    np.savez_compressed(path, **payload)
    return path
