"""Subprocess worker: one image from a local diffusion model, then exit.

Its own process for the reason every model stage here gets one -- the weights
are released when it exits, so the terrain stage and Blender never compete with
a resident diffusion model for graphics memory.

Deliberately model-agnostic: ``AutoPipelineForText2Image`` accepts SDXL-Turbo,
SD 3.5, FLUX and Qwen-Image alike.  A layout map is flat colour blocks, so the
smallest model that follows a palette instruction is the right one; the large
models are worth their memory in M4, not here.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--negative",
        default="photograph, 3d render, perspective, shading, gradient, texture, "
        "text, labels, watermark, blurry edges",
    )
    args = ap.parse_args()

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        sys.exit(
            "the diffusers backend needs torch and diffusers:\n"
            "    pip install 'worldclaw[imagegen]'\n"
            "or use --layout-backend procedural"
        )

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # Slice attention rather than assume headroom: the layout map is generated on
    # the same card that will shortly hold a reconstruction model.
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    kwargs = dict(
        prompt=args.prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
        generator=generator,
    )
    # Turbo/distilled pipelines reject guidance and negative prompts.
    if args.guidance > 0:
        kwargs["guidance_scale"] = args.guidance
        kwargs["negative_prompt"] = args.negative

    image = pipe(**kwargs).images[0]
    image.save(args.out)
    print(f"layout image -> {args.out} ({args.size}x{args.size}, {args.model}, {device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
