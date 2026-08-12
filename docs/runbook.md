# Runbook: a full run on your own machine

## What "full run" means today

**Prompt → plan → layout map → terrain → material → props → renders → checks.**
One sentence in, a walkable rendered world out, everything on local models.

What is **not** in it yet: stage 3's object generation. The ray-pair placement
arithmetic and the depth gate are built and tested (`docs/placement.md`,
`docs/depth-gate.md`), but nothing yet drives SAM3, SAM3D or Hunyuan3D. So the
world comes out with terrain, materials and scattered props — no huts, no
generated trees.

## 0. Environment

`bpy` publishes wheels for **CPython 3.11 only**. On any other version the
terrain stage still runs and you get the height field, OBJ and 16-bit map, but
no renders.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[blender,dev]'
```

## 1. LM Studio

1. Load an instruct model. Anything mid-sized and recent works; it has to fill a
   JSON schema, not know geology. If plans come back thin or repetitive, go up a
   size before changing anything else.
2. Developer tab → **Start Server**. Default `http://localhost:1234/v1`.

## 2. Pre-flight

```bash
worldclaw doctor
```

Checks the Python version, the core libraries, `bpy`, whether the endpoint
answers, whether the model **actually honours a JSON schema** (the planner is
unusable without it), and the optional image-generation stack. Every failure
prints its fix. Non-zero exit means something blocking.

If your model name differs from the default:

```bash
worldclaw llm-check                 # lists what the endpoint serves
worldclaw doctor --llm-model qwen2.5-14b-instruct
```

## 3. The run

```bash
worldclaw all --prompt "Eine ausgetrocknete Schlucht in rotem Sandstein, Geröll am Grund" \
              --llm-model your-model --run rote-schlucht --render
```

Four stages, printed as they go: planning, terrain, Blender, checks. Every stage
is cached on a fingerprint of its inputs, so a failure part-way through costs
only the stage that failed — rerun the same command and it picks up where it
stopped. `--force` recomputes everything.

Same thing stage by stage, when you want to inspect between steps:

```bash
worldclaw plan    --prompt "..." --llm-model your-model --run rote-schlucht
worldclaw terrain --spec artifacts/rote-schlucht/spec/terrain_spec.json --run rote-schlucht
worldclaw blender --run rote-schlucht --render --save-blend
worldclaw check   --run rote-schlucht --spec artifacts/rote-schlucht/spec/terrain_spec.json
```

## 4. What to look at, in order

| File | What it tells you |
|---|---|
| `plan/intent.json` | What the model extracted. Fields the prompt did not state must be `null` — if they are filled, the extraction is inventing. |
| `plan/scene_plan.json` | The design: regions, elevations, character. `inferred_fields` lists what the planner assumed. |
| `plan/layout.png` | The region map. Should be a few clean colour areas. |
| `plan/layout_report.json` | Whether the layout passed the palette gate, and why. |
| `terrain/preview_hillshade.png` | The fastest honest look at the height field. |
| `blender/render_cam_ground.png` | Eye height, standing on the sharpest drop in the scene. This is the believability test. |
| `checks.json` | The acceptance run, machine-readable. |

## 5. Optional: layout map from a real image model

The default `procedural` backend paints the map from a template — no model, no
GPU. To use a local diffusion model instead:

```bash
pip install -e '.[imagegen]'
worldclaw all --prompt "..." --llm-model your-model \
              --layout-backend diffusers --image-model <a small text-to-image model>
```

The model runs in a subprocess that exits afterwards, so it never holds graphics
memory while Blender works. Generated maps are quantised to the palette and
despeckled, and rejected with a retry if the model ignored the palette.

A layout map is the *easy* image task — flat colour blocks, not a
topography-preserving edit. A small model is enough.

## Troubleshooting

**`doctor` says structured output failed.** The planner needs the endpoint to
enforce a JSON schema. Update LM Studio and prefer a model that advertises
structured output or function calling. Very small models often cannot.

**Renders come out black.** Should not happen any more — the two causes are
fixed (a camera far-clip of 100 m at kilometre scale, and a horizon plane above
the viewpoint) and both are pinned by tests. If it recurs, check
`blender/cameras.json` for a camera inside the terrain.

**Renders are slow.** Cycles runs on **CPU on purpose**: the GPU belongs to the
generator models. Lower `--samples`, or `--decimate 2` to halve the mesh.

**The render looks white / colourless, but brightness is normal.**
Check `blender_summary.json > diagnostics > render_stats`. A mean of roughly
120-200 out of 255 means the render is *not* overexposed -- the terrain simply
has no colour to show. That happens when the planner assigns the same category
to several regions (two `rock` regions both get the category's grey) and sets no
`color_hint`. The hint is the only path from the prompt's colour words to the
material: "red sandstone" without it arrives as category `rock` and renders
neutral grey. If a model consistently leaves it null, a larger one usually
fixes it; `plan/scene_plan.json` shows what it chose.

**`scatter_contact` fails.** Props are floating or buried. Usually a scatter
spec with a footprint too large for the local relief — lower
`footprint_radius_m` or raise `embed`.

**The plan is nonsense.** Look at `intent.json` first. If extraction already
invented a biome the prompt never mentioned, the model is ignoring the
extraction instruction and a larger one will help. If extraction is clean but
the plan is odd, that is the planner's design choice — it is allowed to fill
gaps, and it must list them in `inferred_fields`.
