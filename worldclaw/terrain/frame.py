"""The world frame, in one place.

x east, y north, z up; origin at the centre of the world; raster row 0 on the
**north** edge, so the row index runs against world y.

Every consumer -- the OBJ exporter, the Blender builder, the scatter sampler --
converts through this module.  The convention was previously implemented twice
and a test had to pin the copies together; a third copy for scattering would
have been the one to drift.
"""

from __future__ import annotations

import numpy as np


def world_from_raster(
    shape: tuple[int, int], cell_size_m: float, row, col
) -> tuple[np.ndarray, np.ndarray]:
    """Raster indices -> world (x, y) in metres. Accepts scalars or arrays."""
    h, w = shape
    x = (np.asarray(col, dtype=np.float64) - (w - 1) / 2.0) * cell_size_m
    y = ((h - 1) / 2.0 - np.asarray(row, dtype=np.float64)) * cell_size_m
    return x, y


def raster_from_world(
    shape: tuple[int, int], cell_size_m: float, x, y
) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) -> fractional raster indices (row, col).

    Fractional on purpose: the scatter sampler and every contact test need
    sub-cell accuracy, and rounding here is what makes a prop sit a few
    centimetres inside a slope.
    """
    h, w = shape
    col = np.asarray(x, dtype=np.float64) / cell_size_m + (w - 1) / 2.0
    row = (h - 1) / 2.0 - np.asarray(y, dtype=np.float64) / cell_size_m
    return row, col


def grid_extent_m(shape: tuple[int, int], cell_size_m: float) -> tuple[float, float]:
    """Distance between the outermost samples, in metres (x, y)."""
    h, w = shape
    return ((w - 1) * cell_size_m, (h - 1) * cell_size_m)


def sample_bilinear(field: np.ndarray, row, col) -> np.ndarray:
    """Bilinear lookup at fractional raster coordinates, clamped at the border."""
    h, w = field.shape[:2]
    r = np.clip(np.asarray(row, dtype=np.float64), 0.0, h - 1.0)
    c = np.clip(np.asarray(col, dtype=np.float64), 0.0, w - 1.0)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr = (r - r0)[..., None] if field.ndim == 3 else (r - r0)
    fc = (c - c0)[..., None] if field.ndim == 3 else (c - c0)

    top = field[r0, c0] * (1.0 - fc) + field[r0, c1] * fc
    bot = field[r1, c0] * (1.0 - fc) + field[r1, c1] * fc
    return top * (1.0 - fr) + bot * fr


def sample_height_m(height_m: np.ndarray, cell_size_m: float, x, y) -> np.ndarray:
    """Terrain height at world (x, y), bilinearly interpolated."""
    row, col = raster_from_world(height_m.shape, cell_size_m, x, y)
    return sample_bilinear(height_m, row, col)
