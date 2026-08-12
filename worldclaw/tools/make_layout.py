"""Synthetic layout maps -- a stand-in for the hand-painted input of M1.

These paint the same thing a human would with four colours and a brush: coarse
blobs of terrain category, nothing more.  They exist so the terrain stage has a
reproducible input before M3 replaces them with an image model, and so the
regression tests do not depend on a checked-in PNG that nobody can regenerate.

Painting happens at a coarse grid and is upsampled with nearest neighbour, which
keeps the output honestly blocky -- exactly the kind of input the mask stage has
to cope with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..rng import rng as make_rng


def _blob_field(size: int, gen: np.random.Generator, scale: int = 6) -> np.ndarray:
    """Smooth random field in [0, 1] for organic-looking region borders."""
    coarse = gen.random((scale, scale)).astype(np.float32)
    img = Image.fromarray((coarse * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    f = np.asarray(img, dtype=np.float32) / 255.0
    return (f - f.min()) / max(float(np.ptp(f)), 1e-6)


# Blob outline radius = radius_frac * (_BLOB_BASE + _BLOB_AMPL * wobble),
# wobble in [0, 1]. Named so the clearance check cannot drift from the outline.
_BLOB_BASE = 0.5
_BLOB_AMPL = 1.0


def _placed_blob(
    size: int,
    gen: np.random.Generator,
    radius_frac: float,
    count: int = 1,
    where: np.ndarray | None = None,
) -> np.ndarray:
    """One or more organic blobs placed well inside the map.

    Thresholding a random field is the obvious way to get a blob and the wrong
    one: it either clips against the border into a dead-straight edge, or -- if
    the border-touching components are dropped -- silently yields nothing at
    all, leaving a region of the spec unpainted.  Placing the blob and
    perturbing its radius always produces exactly what was asked for.

    ``where`` restricts the centres to a mask -- a massif meant to overlook a
    canyon has to be placed on the plateau, not merely somewhere that is later
    intersected with it, or most of the blob is clipped away.
    """
    yy = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    out = np.zeros((size, size), dtype=bool)

    # The outline reaches radius_frac * (BASE + AMPL), not radius_frac: clearing
    # only the nominal radius lets the blob run off the map and be cut into the
    # dead-straight edge this function exists to avoid.
    max_radius_frac = radius_frac * (_BLOB_BASE + _BLOB_AMPL)
    margin = max_radius_frac

    centres: np.ndarray | None = None
    if where is not None and where.any():
        from scipy import ndimage

        # Pad with background first: distance_transform_edt measures distance to
        # the nearest zero *inside* the array, so a region running to the map
        # edge reports plenty of room right up against it -- and the blob then
        # spills over and is cut straight.
        padded = np.pad(where, 1, constant_values=False)
        dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        room = dist >= max_radius_frac * size
        centres = np.argwhere(room if room.any() else where)

    for i in range(count):
        if centres is not None:
            cy_px, cx_px = centres[gen.integers(len(centres))]
            cy, cx = float(cy_px) / size, float(cx_px) / size
        else:
            cy = float(gen.uniform(margin, 1.0 - margin))
            cx = float(gen.uniform(margin, 1.0 - margin))
        wobble = _blob_field(size, gen, scale=11)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius_frac
        out |= d < (_BLOB_BASE + _BLOB_AMPL * wobble)
    return out


def canyon(size: int, seed: int, colors: dict[str, str]) -> np.ndarray:
    """Plateau split by a meandering canyon, with talus slopes along the rim.

    The strong height contrast is the point: placement errors show up on a
    slope, never on the flat.
    """
    gen = make_rng(seed, "layout", "canyon")
    yy = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]

    # Channel centre meanders north-south down the map.
    centre = 0.5 + 0.16 * np.sin(2.4 * np.pi * yy + gen.random() * 6.28) \
        + 0.05 * np.sin(5.1 * np.pi * yy)
    width = 0.055 + 0.03 * _blob_field(size, gen)[:, :1]
    dist = np.abs(xx - centre)

    labels = np.full((size, size), 0, dtype=np.int32)  # plateau
    labels[dist < width * 2.6] = 1  # slope / talus
    labels[dist < width] = 2  # canyon floor
    # A rock massif so the map is not purely channel-symmetric.  Intersecting
    # with the plateau keeps it from swallowing the channel it should overlook.
    labels[_placed_blob(size, gen, 0.11, where=labels == 0) & (labels == 0)] = 3
    return _paint(labels, ["plateau", "slope", "basin", "rock"], colors)


def desert(size: int, seed: int, colors: dict[str, str]) -> np.ndarray:
    """Near-flat dune sea with gravel flats and a low rock outcrop.

    The flat counterpart to ``canyon``: it exercises the same code path with
    almost no relief, where scale errors are what go wrong instead.
    """
    gen = make_rng(seed, "layout", "desert")
    field = _blob_field(size, gen, scale=5)
    labels = np.full((size, size), 0, dtype=np.int32)  # dune sea
    labels[field < 0.38] = 1  # gravel flat
    labels[_placed_blob(size, gen, 0.075, count=2, where=labels == 0)] = 2  # rock outcrops
    return _paint(labels, ["dune", "sand", "rock"], colors)


def _paint(labels: np.ndarray, order: list[str], colors: dict[str, str]) -> np.ndarray:
    palette = []
    for name in order:
        c = colors[name].lstrip("#")
        palette.append((int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)))
    return np.array(palette, dtype=np.uint8)[labels]


TEMPLATES = {"canyon": canyon, "desert": desert}


def generate(template: str, size: int, seed: int, colors: dict[str, str]) -> np.ndarray:
    try:
        fn = TEMPLATES[template]
    except KeyError:
        raise ValueError(f"unknown layout template {template!r}; have {sorted(TEMPLATES)}") from None
    return fn(size, seed, colors)


def save(rgb: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path
