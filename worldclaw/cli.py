"""Command line entry point.

    worldclaw layout  --template canyon --out layouts/canyon.png
    worldclaw terrain --spec specs/canyon.json --run canyon-01
    worldclaw blender --run canyon-01 --render
    worldclaw run     --spec specs/canyon.json          # all three
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .artifacts import RunContext, new_run_id
from .pipeline import run_terrain_stage
from .schemas import TerrainSpec
from .tools import make_layout


def _load_spec(path: str) -> TerrainSpec:
    return TerrainSpec.model_validate_json(Path(path).read_text())


def cmd_layout(args) -> int:
    spec = _load_spec(args.spec) if args.spec else None
    if spec is not None:
        colors = {r.name: r.color for r in spec.regions}
        size = args.size or spec.resolution
        seed = args.seed if args.seed is not None else spec.seed
    else:
        from .layout.palette import PALETTE

        colors = PALETTE
        size = args.size or 512
        seed = args.seed or 0

    rgb = make_layout.generate(args.template, size, seed, colors)
    out = make_layout.save(rgb, args.out)
    print(f"layout map -> {out}  ({size}x{size}, template={args.template}, seed={seed})")
    return 0


def _resolve_layout(spec: TerrainSpec, args) -> Path:
    """Use the given layout, or synthesise the template the spec names."""
    if args.layout:
        return Path(args.layout)
    if spec.layout_map and Path(spec.layout_map).exists():
        return Path(spec.layout_map)
    template = args.template or spec.layout_template
    if not template:
        raise SystemExit(
            "no layout map: pass --layout, set layout_map or layout_template in "
            "the spec, or pass --template"
        )
    path = Path(spec.layout_map or f"layouts/{spec.name}.png")
    colors = {r.name: r.color for r in spec.regions}
    make_layout.save(make_layout.generate(template, spec.resolution, spec.seed, colors), path)
    print(f"synthesised layout map ({template}) -> {path}")
    return path


def cmd_terrain(args) -> int:
    spec = _load_spec(args.spec)
    layout = _resolve_layout(spec, args)
    run = RunContext(args.run or new_run_id(spec.name), force=args.force)
    art = run_terrain_stage(spec, run, layout_path=layout, obj_stride=args.obj_stride)

    print(run.report())
    print(
        f"  height {art.height_min_m:.1f} .. {art.height_max_m:.1f} m"
        f"   regions: {len(art.region_instances)} instances"
        f"   sha256[:12]={art.checksum[:12]}"
    )
    print(f"  preview: {run.dir / (art.preview_png or '')}")
    return 0


def cmd_blender(args) -> int:
    run_dir = Path("artifacts") / args.run
    hf = run_dir / "terrain" / "heightfield.npz"
    if not hf.exists():
        raise SystemExit(f"no height field in {run_dir}; run `worldclaw terrain` first")

    cmd = [
        sys.executable, "-m", "worldclaw.blender.build_terrain",
        "--heightfield", str(hf),
        "--out-dir", str(run_dir / "blender"),
        "--name", args.name,
        "--decimate", str(args.decimate),
        "--render-samples", str(args.samples),
    ]
    for flag, rel in (("--splat", "terrain/splat.json"), ("--scatter", "terrain/scatter.json")):
        if (run_dir / rel).exists():
            cmd += [flag, str(run_dir / rel)]
    spec_path = run_dir / "spec" / "terrain_spec.json"
    if spec_path.exists():
        cmd += ["--spec", str(spec_path)]
    if args.render:
        cmd.append("--render")
    if not args.no_gltf:
        cmd.append("--export-gltf")
    if args.save_blend:
        cmd.append("--save-blend")
    if args.no_backdrop:
        cmd.append("--no-backdrop")

    # Separate process on purpose: Blender's memory is released before anything
    # else in the pipeline loads.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"blender stage failed ({proc.returncode})")

    summary = run_dir / "blender" / "blender_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        print(
            f"blender {s['blender']}: {s['polygons']} polys, "
            f"{s.get('scatter_instances', 0)} props, {s['world_size_m']:.0f} m across, "
            f"{len(s['renders'])} renders -> {run_dir / 'blender'}"
        )
    return 0


def cmd_check(args) -> int:
    from . import checks

    spec = _load_spec(args.spec)
    run_dir = Path("artifacts") / args.run
    if not (run_dir / "terrain_artifacts.json").exists():
        raise SystemExit(f"no terrain artefacts in {run_dir}; run `worldclaw terrain` first")

    layout = Path(args.layout or spec.layout_map or "")
    results = checks.run_all(run_dir, spec, layout)
    print(f"acceptance checks for {args.run}")
    print(checks.format_report(results))

    (run_dir / "checks.json").write_text(
        json.dumps(
            [{"name": c.name, "status": c.status, "detail": c.detail, "values": c.values}
             for c in results],
            indent=2,
            default=str,
        )
    )
    return 0 if all(c.ok for c in results) else 1


def cmd_run(args) -> int:
    spec = _load_spec(args.spec)
    args.run = args.run or new_run_id(spec.name)
    rc = cmd_terrain(args)
    if rc != 0:
        return rc
    return cmd_blender(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="worldclaw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("layout", help="paint a synthetic layout map")
    p.add_argument("--template", default="canyon", choices=sorted(make_layout.TEMPLATES))
    p.add_argument("--spec", help="take region colours from this spec")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_layout)

    def add_terrain_args(p):
        p.add_argument("--spec", required=True)
        p.add_argument("--layout", help="layout bitmap; defaults to the spec's layout_map")
        p.add_argument("--template", help="synthesise the layout if none exists")
        p.add_argument("--run", help="run id (default: <spec name>-<timestamp>)")
        p.add_argument("--force", action="store_true", help="ignore cached stages")
        p.add_argument("--obj-stride", type=int, default=1)

    p = sub.add_parser("terrain", help="layout map -> height field + meshes")
    add_terrain_args(p)
    p.set_defaults(func=cmd_terrain)

    def add_blender_args(p, with_run=True):
        if with_run:
            p.add_argument("--run", required=True)
        p.add_argument("--name", default="terrain")
        p.add_argument("--render", action="store_true")
        p.add_argument("--samples", type=int, default=24)
        p.add_argument("--decimate", type=int, default=1)
        p.add_argument("--no-gltf", action="store_true")
        p.add_argument("--save-blend", action="store_true")
        p.add_argument("--no-backdrop", action="store_true")

    p = sub.add_parser("blender", help="height field -> .blend/glTF + diagnostic renders")
    add_blender_args(p)
    p.set_defaults(func=cmd_blender)

    p = sub.add_parser("check", help="run the automated acceptance checks")
    p.add_argument("--run", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--layout")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="terrain + blender in one go")
    add_terrain_args(p)
    add_blender_args(p, with_run=False)
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
