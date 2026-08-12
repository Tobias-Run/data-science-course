"""Region weights as splat maps.

``m_tilde_r`` lives in a numpy array, but a Blender shader needs it as an image
and Unreal needs it as landscape layer weights.  Both want the same thing: up to
four regions packed into the RGBA channels of one texture.

Packing rather than one greyscale image per region is not just tidiness -- it is
the format Unreal's landscape layer-weight import already expects, so the M2
material and the M5 landscape read the identical file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CHANNELS = ("R", "G", "B", "A")


def pack_splat_maps(weights: np.ndarray, names: list[str]) -> list[dict]:
    """Pack an (R, H, W) weight stack into RGBA groups of four regions.

    Returns one descriptor per image with the channel-to-region mapping, so a
    consumer never has to guess which channel is which region.
    """
    if weights.shape[0] != len(names):
        raise ValueError("weight stack and region names disagree")

    maps = []
    for start in range(0, len(names), 4):
        group = names[start : start + 4]
        rgba = np.zeros((*weights.shape[1:], 4), dtype=np.float32)
        for c, _ in enumerate(group):
            rgba[..., c] = weights[start + c]
        maps.append(
            {
                "index": start // 4,
                "channels": {CHANNELS[c]: n for c, n in enumerate(group)},
                "rgba": rgba,
            }
        )
    return maps


def write_splat_maps(weights: np.ndarray, names: list[str], out_dir: Path | str) -> list[dict]:
    """Write the packed splat maps as 8-bit PNGs and return their descriptors.

    Eight bits per channel is enough here: these are blend weights, not heights,
    and a 1/255 error in a blend weight is invisible where a 1/255 error in a
    height field would terrace a hillside.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for m in pack_splat_maps(weights, names):
        path = out_dir / f"splat_{m['index']}.png"
        Image.fromarray((np.clip(m["rgba"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGBA").save(
            path
        )
        descriptors.append(
            {"index": m["index"], "path": str(path), "channels": m["channels"]}
        )
    return descriptors
