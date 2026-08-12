# Architecture

## Stages and what flows between them

```
prompt
  │
  ├─ Stage 1  planning      intent → plan → TerrainSpec + layout map     [M3 ✓]
  │            local LLM (schema-constrained) + local image model
  │
  ├─ Stage 2  terrain       masks → weights → height field → mesh,       [M1 ✓ M2 ✓]
  │            splat maps, scattered props                                pure numpy + Blender
  │
  └─ Stage 3  objects       render → edit → segment → reconstruct →      [M4, math ✓]
               place                                                      models + placement
```

Stages communicate **only through files** in `artifacts/<run-id>/`. Never
through shared Python state. Three consequences, all deliberate:

- Every stage is separately inspectable, cacheable and resumable.
- The model-backed stages can run as subprocesses that terminate before the next
  one loads — the VRAM discipline, since SAM3, SAM3D and Hunyuan3D must never be
  resident together.
- A stage can be replaced (a different image model, a different LLM) without the
  others noticing.

## Caching

Each stage declares its inputs and outputs. `RunContext.stage()` fingerprints the
inputs, and skips the body when the fingerprint matches and the outputs exist.
`--force` ignores the cache. Timings land in `manifest.json`, which is also
where the per-stage cost criterion is read from.

## Single sources of truth

The recurring failure mode in a pipeline like this is the same fact implemented
twice and drifting apart. Three places where that was deliberately collapsed:

**The world frame** lives in `terrain/frame.py`. x east, y north, z up, origin at
the world centre, raster row 0 on the **north** edge. The exporter, the Blender
builder and the scatter sampler all convert through it. It was implemented twice
before, with a test pinning the copies together; scattering would have been the
third copy, and the third copy is the one that drifts.

**Region weights** (`m̃_r`) are computed once in `layout/masks.py` and written to
`weights.npz`. Three consumers: the height field, material blending, scatter
density. The paper's "one mask computation" — made literal.

**The contact measurement** is one definition, in `terrain/scatter.py`, applied
to scattered props in M2 and to reconstructed meshes in M4. Same numbers, same
meaning, comparable across milestones.

## Where the numbers are decided

The planner states **intent**, never parameters: "upper plateau, high,
terraced". `planning/expand.py` turns character into noise bands, operator
parameters and scatter densities by rule.

Two reasons. A local model asked for octave counts and erosion iteration budgets
produces confident nonsense. And keeping the mapping in code makes it reviewable
and identical across models — swapping the LLM changes a scene's *design*, never
its numerical sanity.

## Conventions worth knowing before reading the code

| Thing | Convention |
|---|---|
| World frame | x east, y north, z up; origin centred; raster row 0 = north |
| Camera maths | computer-vision: x right, y down, z forward, row 0 top |
| Blender camera | looks down local −z, +y up — converted, never mixed |
| Heights | normalised units in the spec, metres after `height_scale_m` |
| Noise frequency | cycles across the world extent, so specs are resolution-free |
| Terrace steps | per normalised unit; bench height = `height_scale_m / steps` |
| Operators | return a *delta*, scaled by `alpha`; `alpha = 0` is always a no-op |

## Testing strategy

103 tests, about 2.5 seconds, no GPU and no network.

The stages that need models are tested against **scripted stand-ins**:
`ScriptedLLM` replays prepared objects, `ScriptedDepth` replays prepared maps,
and the placement is tested by synthetic round trip. This is not a compromise —
it is what makes the hardest arithmetic in the project debuggable in seconds
instead of minutes with three models resident.

What the tests actually protect is the class of bug the eye cannot catch on a
render: partition of unity, mass conservation in erosion, frame-convention
agreement between two implementations, projection invariance under the contact
search, and byte-level reproducibility from a seed.
