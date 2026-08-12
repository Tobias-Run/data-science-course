"""Layout map -> region masks -> normalised soft weights.

This module is deliberately the *single* place where region weights are
computed.  The same ``m_tilde`` stack drives

  1. the height field   (terrain/heightfield.py),
  2. material assignment and blending   [M2],
  3. asset scattering density           [M2],

which is why it returns the weights as a plain array stack plus the hard masks
and connected components rather than folding them into the terrain code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from ..schemas import TerrainSpec


@dataclass
class RegionMasks:
    """Result of mask extraction for one layout map.

    Attributes
    ----------
    labels : (H, W) int16
        Index into ``names`` for every pixel.
    hard : (R, H, W) bool
        ``hard[r]`` is the crisp painted area of region r.
    weights : (R, H, W) float32
        ``m_tilde``: boundary-softened and normalised, so ``weights.sum(0) == 1``
        everywhere.  This is the partition of unity the height field needs.
    names : list[str]
    unmatched_fraction : float
        Share of pixels whose colour was not an exact palette hit (anti-aliased
        edges of a hand-painted map).  A large value means the palette is wrong.
    """

    labels: np.ndarray
    hard: np.ndarray
    weights: np.ndarray
    names: list[str]
    unmatched_fraction: float

    @property
    def n_regions(self) -> int:
        return len(self.names)

    def weight_of(self, name: str) -> np.ndarray:
        return self.weights[self.names.index(name)]


def load_layout(path: str, resolution: int) -> np.ndarray:
    """Load a layout bitmap as RGB uint8, resampled to ``resolution``.

    Nearest-neighbour resampling on purpose: the map carries categorical colour
    codes, and interpolation would invent colours that match no region.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def quantize_to_palette(rgb: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign every pixel to the nearest palette colour.

    Returns the label image and the fraction of pixels that were not an exact
    match (JPEG artefacts, anti-aliased brush edges -- tolerated, but reported).
    """
    flat = rgb.reshape(-1, 3).astype(np.int32)
    pal = palette.astype(np.int32)
    # (N, R) squared distances; palettes are small, so the full matrix is fine.
    d = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    labels = np.argmin(d, axis=1).astype(np.int16)
    unmatched = float((d[np.arange(len(flat)), labels] > 0).mean())
    return labels.reshape(rgb.shape[:2]), unmatched


def _despeckle(mask: np.ndarray, radius: int) -> np.ndarray:
    """Remove brush speckle: binary opening then closing with a disc."""
    if radius <= 0:
        return mask
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disc = (x * x + y * y) <= radius * radius
    return ndimage.binary_closing(ndimage.binary_opening(mask, disc), disc)


def soft_weights(hard: np.ndarray, blend_px: np.ndarray) -> np.ndarray:
    """Soften region boundaries and normalise to a partition of unity.

    ``m_tilde_r = blur_r(M_r) / sum_s blur_s(M_s)``

    Each region may use its own blend width, so a cliff edge can stay crisp
    while a dune field fades over tens of pixels.  Normalising *after* blurring
    keeps the sum at exactly 1, which is what makes the weighted height sum a
    convex combination of the regional profiles instead of a bump at every
    border.
    """
    blurred = np.empty(hard.shape, dtype=np.float32)
    for r in range(hard.shape[0]):
        m = hard[r].astype(np.float32)
        sigma = float(blend_px[r])
        blurred[r] = ndimage.gaussian_filter(m, sigma, mode="nearest") if sigma > 0 else m

    total = blurred.sum(axis=0)
    # Pixels that no blur reached (empty region set) fall back to the hard mask.
    empty = total <= 1e-8
    if empty.any():
        blurred[:, empty] = hard[:, empty].astype(np.float32)
        total = blurred.sum(axis=0)
        total[total <= 1e-8] = 1.0
    return blurred / total[None, ...]


def extract_masks(spec: TerrainSpec, layout_rgb: np.ndarray) -> RegionMasks:
    """Full mask extraction for a spec and its layout bitmap."""
    palette = np.array([r.rgb for r in spec.regions], dtype=np.uint8)
    labels, unmatched = quantize_to_palette(layout_rgb, palette)

    hard = np.stack([labels == i for i in range(len(spec.regions))])
    if spec.despeckle_px > 0:
        hard = np.stack([_despeckle(m, spec.despeckle_px) for m in hard])
        # Opening can orphan pixels; re-assign them to the nearest surviving region.
        orphan = ~hard.any(axis=0)
        if orphan.any():
            _, (iy, ix) = ndimage.distance_transform_edt(orphan, return_indices=True)
            for r in range(hard.shape[0]):
                hard[r][orphan] = hard[r][iy[orphan], ix[orphan]]
        labels = np.argmax(hard, axis=0).astype(np.int16)

    blend = np.array(
        [r.blend_width_px if r.blend_width_px is not None else spec.blend_width_px for r in spec.regions],
        dtype=np.float32,
    )
    weights = soft_weights(hard, blend)
    return RegionMasks(
        labels=labels,
        hard=hard,
        weights=weights.astype(np.float32),
        names=[r.name for r in spec.regions],
        unmatched_fraction=unmatched,
    )


def connected_components(mask: np.ndarray, min_area_px: int = 64) -> list[np.ndarray]:
    """Split one region's hard mask into its painted patches.

    Stage 3 populates *instances* -- a single clearing, a single canyon floor --
    so the terrain stage already reports them.
    """
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area_px:
            out.append(comp)
    return out


def colorize_labels(labels: np.ndarray, spec: TerrainSpec) -> np.ndarray:
    """Render the quantised label image back to RGB for inspection."""
    palette = np.array([r.rgb for r in spec.regions], dtype=np.uint8)
    return palette[np.clip(labels, 0, len(spec.regions) - 1)]
