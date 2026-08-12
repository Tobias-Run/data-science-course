"""Layout map generation, and the gate that makes a generated one usable.

An image model does not emit exact palette colours -- it emits gradients,
anti-aliased edges and colours that are merely close.  That is fine: the map is
categorical, so the pipeline quantises to the nearest palette entry and cleans
up the speckle.  Both steps already existed for hand-painted input.

What matters is measuring whether the result is *usable*, which is what
``evaluate_layout`` does.  ``unmatched_fraction`` becomes the acceptance metric
for the image model rather than a reason to avoid one: a model that renders the
requested colours faithfully scores near zero, one that invents its own palette
scores high and the layout is rejected before it can produce a nonsense terrain.

Backends
--------
``procedural``  the template painter from M1; no model, always available.
``comfyui``     an already-running ComfyUI instance over HTTP.
``diffusers``   a local diffusion model in a subprocess that exits afterwards,
                which is the same VRAM discipline the Blender stage follows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..layout.masks import quantize_to_palette
from ..schemas import TerrainSpec

PALETTE_INSTRUCTION = (
    "Top-down categorical region map, flat solid colour areas with hard edges, "
    "no shading, no gradients, no texture, no text, no labels, no perspective, "
    "no lighting. Use exactly these colours and no others: {colours}."
)


def palette_prompt(spec: TerrainSpec, scene_prompt: str) -> str:
    """Compose the image prompt, naming the exact colours the map must use."""
    colours = ", ".join(f"{r.color} for {r.name.replace('_', ' ')}" for r in spec.regions)
    return f"{scene_prompt.strip()} {PALETTE_INSTRUCTION.format(colours=colours)}"


@dataclass
class LayoutReport:
    """How well a generated map fits the spec's palette."""

    unmatched_fraction: float
    coverage: dict[str, float]
    missing_regions: list[str]
    accepted: bool
    reason: str

    def model_dump(self) -> dict:
        return {
            "unmatched_fraction": self.unmatched_fraction,
            "coverage": self.coverage,
            "missing_regions": self.missing_regions,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def evaluate_layout(
    rgb: np.ndarray,
    spec: TerrainSpec,
    max_unmatched: float = 0.35,
    min_region_share: float = 0.005,
) -> LayoutReport:
    """Quantise to the palette and judge whether the result is usable.

    ``max_unmatched`` is generous on purpose: anti-aliased edges alone put a
    perfectly good generated map in the 10-25% range, because every boundary
    pixel is a blend of two palette colours.  What it catches is a model that
    ignored the palette entirely.
    """
    palette = np.array([r.rgb for r in spec.regions], dtype=np.uint8)
    labels, unmatched = quantize_to_palette(rgb, palette)
    coverage = {r.name: float((labels == i).mean()) for i, r in enumerate(spec.regions)}
    missing = [n for n, c in coverage.items() if c < min_region_share]

    if unmatched > max_unmatched:
        return LayoutReport(unmatched, coverage, missing, False,
                            f"palette mismatch {unmatched:.1%} > {max_unmatched:.0%}")
    if missing:
        return LayoutReport(unmatched, coverage, missing, False,
                            f"regions absent from the map: {missing}")
    return LayoutReport(unmatched, coverage, missing, True, "accepted")


def clean_layout(rgb: np.ndarray, spec: TerrainSpec, smooth_px: int = 3) -> np.ndarray:
    """Snap a generated image onto the palette and remove speckle.

    A diffusion model leaves isolated stray pixels and ragged one-pixel fringes.
    Left in place they become thousands of single-cell regions, and the mask
    stage would faithfully turn each one into its own landform.
    """
    palette = np.array([r.rgb for r in spec.regions], dtype=np.uint8)
    labels, _ = quantize_to_palette(rgb, palette)
    if smooth_px > 0:
        # Majority filter: each pixel takes the most common label in its
        # neighbourhood, which removes speckle without moving real borders.
        stack = np.stack(
            [
                ndimage.uniform_filter((labels == i).astype(np.float32), size=smooth_px)
                for i in range(len(spec.regions))
            ]
        )
        labels = np.argmax(stack, axis=0).astype(np.int16)
    return palette[labels]


# ---------------------------------------------------------------- backends


def _run_procedural(spec: TerrainSpec, template: str, out: Path) -> np.ndarray:
    from ..tools import make_layout

    # Templates address fixed roles; a planned spec names its regions freely, so
    # the two are matched by elevation inside generate_for_spec.
    rgb = make_layout.generate_for_spec(template, spec.resolution, spec.seed, spec)
    make_layout.save(rgb, out)
    return rgb


def _run_comfyui(prompt: str, spec: TerrainSpec, out: Path, endpoint: str, workflow: Path) -> np.ndarray:
    """Drive an already-running ComfyUI instance.

    ComfyUI keeps the model resident, which is the opposite of this pipeline's
    VRAM discipline -- so it is offered, not preferred, and is the right choice
    only when it is already part of the user's setup.
    """
    import urllib.request

    wf = json.loads(Path(workflow).read_text())
    # Substitute the prompt into any node that declares a positive text input.
    for node in wf.values():
        if isinstance(node, dict) and "inputs" in node:
            if "text" in node["inputs"] and node["inputs"].get("_worldclaw") != "negative":
                node["inputs"]["text"] = prompt
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        json.loads(resp.read())
    raise NotImplementedError(
        "ComfyUI submission succeeded but result retrieval depends on the "
        "workflow's SaveImage node; point --layout-out at its output directory "
        "or use the diffusers backend."
    )


def _run_diffusers(prompt: str, spec: TerrainSpec, out: Path, model_id: str,
                   steps: int, guidance: float) -> np.ndarray:
    """Run a local diffusion model in a subprocess that exits when done."""
    script = Path(__file__).with_name("_diffusers_worker.py")
    cmd = [
        sys.executable, str(script),
        "--prompt", prompt,
        "--out", str(out),
        "--model", model_id,
        "--size", str(spec.resolution),
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(spec.seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"diffusers backend failed ({proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return np.asarray(Image.open(out).convert("RGB"))


def generate_layout(
    spec: TerrainSpec,
    prompt: str,
    out_path: Path | str,
    backend: str = "procedural",
    template: str = "canyon",
    model_id: str = "stabilityai/sdxl-turbo",
    steps: int = 8,
    guidance: float = 2.0,
    comfy_endpoint: str = "http://localhost:8188",
    comfy_workflow: Path | str | None = None,
    max_attempts: int = 3,
) -> tuple[np.ndarray, LayoutReport]:
    """Produce a layout map and gate it against the spec's palette.

    Retries a failing generation up to ``max_attempts`` -- the same shape as the
    paper's agentic loop, with the budget enforced in code rather than trusted
    to a prompt.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last: LayoutReport | None = None

    for attempt in range(max_attempts):
        if backend == "procedural":
            rgb = _run_procedural(spec, template, out_path)
        elif backend == "diffusers":
            rgb = _run_diffusers(prompt, spec, out_path, model_id, steps, guidance)
        elif backend == "comfyui":
            if comfy_workflow is None:
                raise ValueError("comfyui backend needs --comfy-workflow")
            rgb = _run_comfyui(prompt, spec, out_path, comfy_endpoint, Path(comfy_workflow))
        else:
            raise ValueError(f"unknown layout backend {backend!r}")

        report = evaluate_layout(rgb, spec)
        last = report
        if report.accepted or backend == "procedural":
            cleaned = clean_layout(rgb, spec)
            Image.fromarray(cleaned).save(out_path)
            return cleaned, report

    assert last is not None
    return np.asarray(Image.open(out_path).convert("RGB")), last
