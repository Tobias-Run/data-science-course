"""Colour codes for hand-painted layout maps.

The layout map is a categorical bitmap: each colour is a terrain category.  The
authoritative palette is whatever the ``TerrainSpec`` regions declare; these
names exist so a human (or, from M3, the image model) paints with a known set,
and so the synthetic layout generator has something to draw with.

Colours are chosen far apart in RGB so nearest-colour quantisation survives
JPEG compression and anti-aliased brush edges.
"""

from __future__ import annotations

PALETTE: dict[str, str] = {
    # water and low ground
    "water": "#1f6fb4",
    "basin": "#2f9e8f",
    "plain": "#4c9a2a",
    "sand": "#e0c072",
    # mid ground
    "dune": "#c8894b",
    "slope": "#a2704a",
    "forest": "#1e5b2e",
    # high ground
    "plateau": "#b0a08c",
    "rock": "#8c8c8c",
    "peak": "#e8e8e8",
    "cliff": "#4a4a4a",
}


def rgb(name: str) -> tuple[int, int, int]:
    c = PALETTE[name].lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def min_separation() -> float:
    """Smallest Euclidean RGB distance in the palette.

    Sanity guard for anyone adding a colour: quantisation gets unreliable once
    two categories sit closer than roughly one JPEG block of noise.
    """
    items = [rgb(n) for n in PALETTE]
    best = float("inf")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = sum((a - b) ** 2 for a, b in zip(items[i], items[j])) ** 0.5
            best = min(best, d)
    return best
