"""Stage 2 driver: layout map in, terrain artefacts out.

Each step is a cached stage, so re-running after a spec tweak only recomputes
what the tweak touched.  The stage boundaries are also the process boundaries we
will need later: from M4 the model-backed steps run as subprocesses that
terminate before the next one loads, and they exchange exactly these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import RunContext, sha256_bytes
from .layout import masks as masks_mod
from .schemas import TerrainArtifacts, TerrainSpec
from .terrain import heightfield as hf_mod
from .terrain import mesh as mesh_mod


def run_terrain_stage(
    spec: TerrainSpec,
    run: RunContext,
    layout_path: str | Path | None = None,
    obj_stride: int = 1,
) -> TerrainArtifacts:
    layout_path = Path(layout_path or spec.layout_map or "")
    if not layout_path.exists():
        raise FileNotFoundError(f"layout map not found: {layout_path}")

    spec_path = run.write_model(spec, "spec", "terrain_spec.json")
    layout_hash = sha256_bytes(layout_path.read_bytes())

    weights_npz = run.path("terrain", "weights.npz")
    labels_png = run.path("terrain", "labels.png")
    masks_json = run.path("terrain", "masks.json")

    # ---- masks + shared soft weights -------------------------------------
    inputs = {"spec": spec.model_dump(mode="json"), "layout": layout_hash}
    with run.stage("masks", inputs, [weights_npz, labels_png, masks_json]) as todo:
        if todo:
            layout_rgb = masks_mod.load_layout(str(layout_path), spec.resolution)
            rm = masks_mod.extract_masks(spec, layout_rgb)
            np.savez_compressed(
                weights_npz, weights=rm.weights, hard=rm.hard, labels=rm.labels
            )
            from PIL import Image

            Image.fromarray(masks_mod.colorize_labels(rm.labels, spec)).save(labels_png)
            run.write_json(
                {
                    "names": rm.names,
                    "unmatched_fraction": rm.unmatched_fraction,
                    "coverage": {
                        n: float(rm.hard[i].mean()) for i, n in enumerate(rm.names)
                    },
                    "partition_max_error": float(
                        np.abs(rm.weights.sum(axis=0) - 1.0).max()
                    ),
                },
                "terrain",
                "masks.json",
            )

    with np.load(weights_npz) as z:
        rm = masks_mod.RegionMasks(
            labels=z["labels"],
            hard=z["hard"],
            weights=z["weights"],
            names=[r.name for r in spec.regions],
            unmatched_fraction=json.loads(masks_json.read_text())["unmatched_fraction"],
        )

    # ---- height field -----------------------------------------------------
    height_npz = run.path("terrain", "heightfield.npz")
    stats_json = run.path("terrain", "terrain_stats.json")
    with run.stage("heightfield", inputs, [height_npz, stats_json]) as todo:
        if todo:
            hf = hf_mod.build_heightfield(spec, rm)
            mesh_mod.save_npz(hf, height_npz, weights=rm.weights)
            run.write_json(hf.stats(), "terrain", "terrain_stats.json")

    with np.load(height_npz) as z:
        hf = hf_mod.Heightfield(
            height_m=z["height_m"], normalised=z["normalised"], spec=spec
        )

    # ---- exports ----------------------------------------------------------
    obj_path = run.path("terrain", "terrain.obj")
    png16_path = run.path("terrain", "heightmap_16bit.png")
    preview_path = run.path("terrain", "preview_hillshade.png")
    instances_json = run.path("terrain", "region_instances.json")
    export_inputs = {**inputs, "obj_stride": obj_stride}

    with run.stage(
        "terrain_export", export_inputs, [obj_path, png16_path, preview_path, instances_json]
    ) as todo:
        if todo:
            mesh_mod.write_obj(hf, obj_path, stride=obj_stride)
            _, rng_info = mesh_mod.write_heightmap_png16(hf, png16_path)
            mesh_mod.write_preview_png(hf, preview_path)
            instances = hf_mod.region_instances(spec, rm, hf)
            run.write_json(
                {
                    "heightmap_range": rng_info,
                    "instances": [i.model_dump(mode="json") for i in instances],
                },
                "terrain",
                "region_instances.json",
            )

    instances = [
        hf_mod.RegionInstance(**i) for i in json.loads(instances_json.read_text())["instances"]
    ]
    stats = json.loads(stats_json.read_text())

    art = TerrainArtifacts(
        spec_name=spec.name,
        run_id=run.run_id,
        resolution=spec.resolution,
        world_size_m=spec.world_size_m,
        height_scale_m=spec.height_scale_m,
        height_min_m=stats["height_min_m"],
        height_max_m=stats["height_max_m"],
        heightfield_npz=run.rel(height_npz),
        weights_npz=run.rel(weights_npz),
        labels_png=run.rel(labels_png),
        heightmap_png16=run.rel(png16_path),
        mesh_obj=run.rel(obj_path),
        preview_png=run.rel(preview_path),
        region_instances=instances,
        checksum=sha256_bytes(hf.height_m.tobytes()),
    )
    run.write_model(art, "terrain_artifacts.json")
    run.manifest["spec"] = run.rel(spec_path)
    run.manifest["layout_sha256"] = layout_hash
    run.save_manifest()
    return art
