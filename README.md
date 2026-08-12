# WorldClaw — Nachbau

Reimplementation of the pipeline described in Tencent Hunyuan, *"WorldClaw:
Agentic 3D Open-World Generation at Scale"* (arXiv 2608.05248). The report ships
no code; everything here is written from the paper's description.

**Status: M1, M2 and M3 complete; M4's arithmetic is done and tested.** A layout map goes in, an explicit walkable,
textured, populated terrain comes out — height field, region weights, splat
maps, blended procedural material, scattered props with a measured contact rate,
OBJ, 16-bit height map, glTF, and diagnostic renders from headless Blender. The schemas for the later stages are
already declared, so the terrain stage writes its artefacts in the shape stage 3
will read them.

```
Prompt q
  ├─ Stage 1  Intent + planning        -> SceneSpec, TerrainSpec, layout map   [M3 ✓]
  ├─ Stage 2  Global terrain           -> height field, weights, meshes,
  │                                        splat maps, scattered props        [M1 ✓ M2 ✓]
  └─ Stage 3  Regional objects         -> PlacementRecord per object           [M4:
                                           ray-pair placement ✓, depth gate ✓,
                                           models pending]
```

## Planning from a prompt (M3)

```bash
worldclaw llm-check                                   # is LM Studio reachable?
worldclaw plan --prompt "A dried-out canyon in red sandstone, scree on the floor" \
               --llm-model your-model --run red-gorge
worldclaw terrain --spec artifacts/red-gorge/spec/terrain_spec.json --run red-gorge
worldclaw blender --run red-gorge --render
```

**Runs entirely on local models.** The planner talks to any OpenAI-compatible
endpoint — LM Studio, Ollama, llama.cpp's server, vLLM — over the standard
library, no vendor SDK. Default `http://localhost:1234/v1`, LM Studio's.

**Schema-constrained decoding.** The endpoint receives a JSON schema and the
model physically cannot emit anything else, so the planner cannot return
something the pipeline fails to parse. `$ref`/`$defs` are inlined first: grammar
compilers vary in how well they follow references, and a half-followed reference
degrades silently into an unconstrained model.

### Intent and planning are separate, as in the paper

Intent analysis **extracts only what the prompt states** — an unmentioned field
stays `null`. Planning then fills the gaps and lists every field it invented in
`inferred_fields`, so what the user asked for stays distinguishable from what
was assumed. Merged into one step, a confident model quietly overwrites the
user's intent and nothing afterwards can tell the difference.

### The planner states intent, not parameters

It never emits noise frequencies or erosion iteration counts. It says *"upper
plateau, high, terraced"* and `planning/expand.py` turns character into numbers
by rule. Two reasons: a local model asked for octave counts produces confident
nonsense, and the numbers stay reviewable and identical across models — swapping
the LLM changes a scene's design, never its numerical sanity. One rule earns its
keep visibly: terrace step count is derived from the scene's relief, so benches
stay around 12 m whether the scene is a dune field or a mountain range.

### The layout map

The planner writes an image prompt; a **local** image model draws it. Backends:
`procedural` (the M1 template painter, no model, always available),
`diffusers` (a local diffusion model in a subprocess that exits afterwards —
the same VRAM discipline as the Blender stage) and `comfyui`.

A diffusion model does not emit exact palette colours, and it does not need to.
The map is categorical, so the output is quantised to the nearest palette entry
and despeckled — machinery that already existed for hand-painted input. What
guards it is `evaluate_layout`: a model that renders the requested colours
scores near zero unmatched, one that painted a landscape instead is rejected
before it can produce a nonsense terrain, and the generation retries within a
budget enforced in code.

Making that gate real required fixing the metric it rests on. `unmatched_fraction`
counted only *exact* palette hits, which is right for a painted PNG and useless
for anything generated: one step of noise put it at 100%. It is now a distance,
with the tolerance derived from the palette's own minimum separation (capped, or
a sparse palette would accept mid-grey as a match for either of two colours).

A layout map is the *easy* image task — flat colour blocks, not a
topography-preserving edit. A small local model is enough here; the graphics card
gets tight in M4, not in M3.

## Quick start

```bash
pip install -e '.[blender,dev]'          # bpy needs CPython 3.11

worldclaw run --spec specs/canyon.json --run canyon-01 --render
worldclaw check --run canyon-01 --spec specs/canyon.json
```

`run` synthesises the layout map (the spec names a template), builds the terrain,
and drives Blender. Everything lands in `artifacts/<run-id>/`:

```
spec/terrain_spec.json      the exact spec this run used
terrain/labels.png          layout map after colour quantisation
terrain/weights.npz         m̃_r — the soft region weights, shared by three consumers
terrain/heightfield.npz     H(x) in metres, the artefact every later stage reads
terrain/heightmap_16bit.png Unreal landscape import path (M5)
terrain/terrain.obj         triangle mesh
terrain/preview_hillshade.png
terrain/region_instances.json  per-patch area/height/slope — stage 3 picks from here
terrain/splat/splat_0.png   RGBA blend weights; also the Unreal layer-weight format
terrain/splat.json          which channel is which region
terrain/scatter.json        placed props + per-prop contact ratio, gap, penetration
blender/cameras.json        K and E per view: stage 3 back-projects through these
blender/render_*.png        overview, oblique, and an eye-height view on the rim
manifest.json               per-stage input fingerprints, outputs and timings
```

Individual stages, all cached on their input fingerprint (`--force` to ignore):

```bash
worldclaw layout  --template canyon --spec specs/canyon.json --out layouts/canyon.png
worldclaw terrain --spec specs/canyon.json --run canyon-01
worldclaw blender --run canyon-01 --render --save-blend
```

Two fixed test prompts, per the acceptance criteria — one flat, one with strong
relief, because placement errors show up on a slope and never on the flat:
`specs/desert.json` and `specs/canyon.json`.

## Materials and scattering (M2)

**The weights become a splat map.** ``m_tilde_r`` is packed four regions to an
RGBA PNG. A Blender shader needs the weights as an image and Unreal needs them
as landscape layer weights; both read the identical file. The shader mixes base
colour and roughness per channel and then lays a slope-driven rock layer over the
top, so steep ground shows rock whatever the region says. The material is purely
procedural, so glTF carries geometry and UVs but not this material — M5 rebuilds
an equivalent landscape material from the same splat map.

**Scattering, and why the contact test lives here.** Candidates are drawn by
affinity (the soft weights) at the spec's target density, filtered by slope and
height, thinned to a minimum spacing, then given a scale, a yaw and an axis
blended between world up and the surface normal.

M2's acceptance criterion ("props without floating or interpenetration") and the
paper's contact-rate metric are the same measurement, so it is implemented once,
against the terrain, and M4 reuses it for reconstructed meshes. Two details it
took a wrong answer to find:

* Contact is measured against the prop's **base plane**, which is perpendicular
  to its blended axis — not against a horizontal slab. A prop tilted onto a
  slope has a tilted base; scoring it horizontally invents interpenetration and
  dragged the canyon contact rate down to 84%. Measuring the real plane puts it
  at 99.7%.
* The base is anchored low in the footprint's height distribution and sunk by a
  fraction of the local relief. Sitting a prop at its centre height leaves it
  floating on the downhill side of any slope.

Reported per prop and aggregated: contact ratio, largest gap, largest
penetration. Gap and penetration are separate because they need opposite fixes.

## The height field

```
H(x) = Σ_r  m̃_r(x) · [ h_r + Σ_k w_rk · N_rk(x) + Σ_j α_rj · G_rj(x) ]
```

The layout map is a categorical bitmap: each colour is a terrain category. Masks
are extracted by nearest-colour quantisation, optionally despeckled, blurred with
a per-region blend width, and normalised into `m̃_r` — a partition of unity, so
`H` is a convex combination of the regional profiles rather than a sum with a
step at every border. That normalisation is checked, not assumed
(`partition_of_unity`).

**One mask computation, three consumers.** The same `m̃_r` stack drives the height
field (M1), material assignment and blending (M2), and scatter density (M2). It
lives in `layout/masks.py` and is written to `weights.npz`; nothing recomputes it.

Regional profiles are evaluated over the whole raster and only then weighted, so
a landform fades across its border instead of being clipped at it. Operators see
the regional base *including* its noise — that is what makes terrace and erosion
meaningful; they modify a profile rather than invent one.

### Noise

Gradient (Perlin) noise, not value noise: value noise puts its extrema on the
lattice points, and terracing downstream turns that into a visible rectangular
maze. Each octave is rotated by the golden angle so stacked octaves never come
back into axis alignment. Frequencies are in cycles across the world extent, so
a spec produces the same landform at any raster resolution.

### Geomorphological operators

The paper names peak / dune / terrace / erosion but does not formulate them.
These are our own formulations, so **expect divergence from the published
figures** — a deliberate, accepted deviation. Each operator returns a *delta* to
be scaled by `α_rj`, which is what lets purely modifying operators live in the
same additive sum as generative ones; `α = 0` is always a no-op.

| Operator | Formulation |
|---|---|
| `peak` | `cos(½π·d/r)^s` bumps at centres sampled from the region's interior (distance transform, so a summit is never cut in half by the border), merged with `max` rather than summed so neighbouring peaks form a ridge instead of stacking. Optional ridged-noise modulation carves radial gullies. |
| `dune` | `sin(φ + a·sin φ)` along a wind direction — the skew is the windward/lee asymmetry of a migrating dune. Mapped to [0,1] (dunes add relief, never dig), sharpened by an exponent, and phase-warped by low-frequency noise so crest lines meander instead of running dead straight. |
| `terrace` | Quantise the base into `steps` benches with a soft step `t^p/(t^p+(1−t)^p)`, `p = 1/(1−sharpness)`, and return `quantised − base`. Step edges are noise-warped. Sharpness 0 is the identity. |
| `erosion` | Thermal relaxation towards the angle of repose: material above the talus gradient slides to lower neighbours, distributed proportionally over the four neighbours. The talus threshold converts the repose angle through the cell size, so the same spec erodes consistently at any resolution. Unconditionally stable for `strength ≤ 1` and mass-conserving to machine precision (tested). |

Terrace steps are counted per *normalised* height unit, so bench height in metres
is `height_scale_m / steps` — worth remembering when a plateau comes out dead
flat because its noise amplitude is smaller than one step.

## World frame

Fixed once and relied on everywhere: **x east, y north, z up**, origin at the
centre of the world, raster row 0 on the **north** edge. Two consequences that
have already cost bugs and are now covered by tests:

- `d/dy = −d/drow`. Getting the sign wrong flips the lighting north-south and
  makes every hill render as a pit.
- The exporter (`terrain/mesh.py`) and the Blender builder
  (`blender/build_terrain.py`) implement this convention independently;
  `test_world_frame_matches_the_blender_stage` pins them together, because stage
  3 back-projects rays through it.

## Blender

Blender is the workshop, Unreal the shop window. The Blender stage runs as its
own process (`python -m worldclaw.blender.build_terrain`) and exits when done —
not tidiness but the VRAM strategy: no generator model and no Blender session
ever hold graphics memory at the same time, and every hand-off goes through files
in the run directory. Cycles renders on CPU for the same reason.

Cameras are aimed by construction rather than by aim-constraints, and their `K`
and `E_world_to_cam` are written to `cameras.json` with every render — stage 3
needs exactly the intrinsics and extrinsics of the image it edits, and recovering
them later from a `.blend` would be guesswork. Two things that are easy to get
wrong and are handled explicitly: Blender's default 100 m far clip silently culls
a kilometre-scale terrain and renders bare sky, and a sun lamp plus a physical
sky at default strengths renders the terrain as a featureless white sheet.

The eye-height camera stands on the point of maximum *local relief* — the canyon
rim, not the highest summit — and looks across the drop. It answers the M1
question (is this walkable, is the scale believable) and is the camera type stage
3 will render for regional composition.

## Acceptance checks

`worldclaw check` exits non-zero on failure, so it works as a gate rather than a
dashboard. The paper evaluates purely qualitatively; these are ours.

Implemented: region coverage (an unpainted region is otherwise silent — this
caught a layout generator that dropped two regions while the pipeline reported
success), partition of unity, finite height field, slope plausibility, 16-bit
round-trip, byte-level reproducibility from the seed, splat partition after 8-bit
quantisation, scatter contact rate, per-stage timing.

Reported as `skipped` with the milestone that will implement them: scale
plausibility, topography fidelity of the image edit, cost per scene. They
are listed rather than omitted so the report always shows the full criterion set.

```bash
python -m pytest          # 63 tests, ~2 s
```

## What the official material adds

The paper's [GitHub repository](https://github.com/Tencent-Hunyuan/Hunyuan3D-WorldClaw)
is a placeholder (README + figures, no code, no licence, no release timeline as
of 2026-08-12), but its pipeline figure is unusually detailed and pins down
several things the text leaves open. Incorporated here:

- **Iteration budgets.** The figure shows `/loop (3/5)` on stage-2 scene
  refinement and `/loop (6/15)` on stage-3 per-region refinement — so the hard
  budgets are ~5 per scene and ~15 per region. `RegionalPlan.iteration_budget`
  defaults to 15 accordingly.
- **Population zones are planned, not derived.** Stage 3 regions are semantic
  windows the planner draws onto the layout map ("Cabin Village", "Fishing
  Village"), not connected components of terrain categories. `RegionalPlan`
  carries a `bbox_px` window for this; `RegionInstance` remains the
  terrain-category view.
- **~50 detected objects per region** (`obj 1 … obj 50` in the detection box);
  `RegionalPlan.max_objects` defaults to 50.
- **Architecture confirmation.** Their agent drives per-stage scripts over YAML
  plans (`terrain_param.yaml`, `scatter_plan.yaml`, `img_gen.py`, `terrain.py`,
  `scatter.py`, `img_edit.py`, `3d_placement.py`) — the same
  files-plus-small-CLIs shape as this repo, which was a guess and is now a match.
  Texture generation is explicitly "procedural node graph OR texture map", as
  planned for M2, and the final scene ships normal + instance maps.

Not recoverable from the figure: the §3.2 tolerances `[-ε⁻, +ε⁺]`, the contact
threshold, and the operator formulations. Those need the full text (blocked from
this environment's network) or remain our own choices.

## Documentation

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Stage boundaries, caching, conventions, testing strategy |
| [docs/placement.md](docs/placement.md) | Ray-pair object placement, paper section 3.2, in full |
| [docs/depth-gate.md](docs/depth-gate.md) | Topography fidelity of the image edit, and its known limitation |
| [docs/operators.md](docs/operators.md) | The four geomorphological operators, as we formulated them |
| [docs/overview.html](docs/overview.html) | A readable single-page overview (German), also published as an artifact |

## Deviations from the paper

| | Paper | Here |
|---|---|---|
| Hardware | 4× H20, parallel | one GPU, strictly sequential, models unloaded between stages |
| Agent | Claude Opus 4.8 | Claude Code as orchestrator |
| Engine | Blender 5.1.1 + BlenderMCP | Blender 5.0 headless via `bpy` |
| Operators | named only | own formulations, documented above |
| PBR textures | 2048² / 1024² | 1024² / 512² planned for M2 |

## Next

- **M4** — render → image edit → SAM3 → SAM3D → ray-pair placement (§3.2), with a
  depth-map gate on the edit: if the image model moves the topography, the ray
  back-projection is invalid and the edit must be rejected.
- **M5** — both refinement loops with hard iteration budgets enforced in code,
  then the Unreal export path.
