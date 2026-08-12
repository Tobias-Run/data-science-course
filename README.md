# WorldClaw — Nachbau

Reimplementation of the pipeline described in Tencent Hunyuan, *"WorldClaw:
Agentic 3D Open-World Generation at Scale"* (arXiv 2608.05248). The report ships
no code; everything here is written from the paper's description.

**Status: M1 complete.** A layout map goes in, an explicit walkable terrain mesh
comes out — height field, region weights, OBJ, 16-bit height map, glTF, and
diagnostic renders from headless Blender. The schemas for the later stages are
already declared, so the terrain stage writes its artefacts in the shape stage 3
will read them.

```
Prompt q
  ├─ Stage 1  Intent + planning        -> SceneSpec, TerrainSpec, layout map   [M3]
  ├─ Stage 2  Global terrain           -> height field, weights, meshes        [M1 ✓ / M2]
  └─ Stage 3  Regional objects         -> PlacementRecord per object           [M4]
```

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
round-trip, byte-level reproducibility from the seed, per-stage timing.

Reported as `skipped` with the milestone that will implement them: contact rate,
scale plausibility, topography fidelity of the image edit, cost per scene. They
are listed rather than omitted so the report always shows the full criterion set.

```bash
python -m pytest          # 29 tests, ~1 s
```

## Deviations from the paper

| | Paper | Here |
|---|---|---|
| Hardware | 4× H20, parallel | one GPU, strictly sequential, models unloaded between stages |
| Agent | Claude Opus 4.8 | Claude Code as orchestrator |
| Engine | Blender 5.1.1 + BlenderMCP | Blender 5.0 headless via `bpy` |
| Operators | named only | own formulations, documented above |
| PBR textures | 2048² / 1024² | 1024² / 512² planned for M2 |

## Next

- **M2** — procedural PBR node graphs per region, blended along `m̃_r`; scatter
  sampling filtered by height, slope and normal. The weights and the surface
  normals both already exist and are tested.
- **M3** — intent analysis (extracts only what the prompt states) and planning
  (fills the gaps), then the layout map from an image model. `SceneSpec` records
  which fields the planner inferred, so user intent stays distinguishable from
  defaults.
- **M4** — render → image edit → SAM3 → SAM3D → ray-pair placement (§3.2), with a
  depth-map gate on the edit: if the image model moves the topography, the ray
  back-projection is invalid and the edit must be rejected.
- **M5** — both refinement loops with hard iteration budgets enforced in code,
  then the Unreal export path.
