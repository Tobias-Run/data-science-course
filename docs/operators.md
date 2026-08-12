# Geomorphological operators

The paper names peak / dune / terrace / erosion but does not formulate them.
These are our own formulations, and the briefing's risk table accepts the
resulting divergence from the published figures as deliberate. This document is
that acceptance, written down.

## The shared contract

```
H(x) = Σ_r  m̃_r(x) · [ h_r + Σ_k w_rk · N_rk(x) + Σ_j α_rj · G_rj(x) ]
```

Every operator receives the regional field **including its noise** and returns a
**delta** in normalised height units, roughly in [-1, 1], to be scaled by `α_rj`.

Returning a delta is what lets purely modifying operators (terrace, erosion)
live in the same additive sum as generative ones (peak, dune). `α` then reads as
"how much of this modification", and `α = 0` is always a no-op — asserted by
test, because an operator that does something at zero weight would make every
spec's numbers lie.

Operators see the noise because that is what makes terrace and erosion
meaningful: they modify a profile rather than invent one.

## peak — isolated summits

`cos(½π · d/r)^s` bumps, giving value 1 and zero gradient at the summit and C1
continuity at the foot.

- Centres are sampled from the region's **interior** via a distance transform, so
  a summit is never bisected by the weight normalisation at a border.
- Bumps merge with `max`, not a sum. Summing stacks neighbouring peaks into one
  implausible mass; taking the maximum makes them form a ridge line.
- Optional ridged-noise modulation carves radial gullies down the flanks.

The cosine is clipped at zero: `cos(π/2)` evaluates a hair *below* zero in
float32, and a fractional `sharpness` on a negative base is NaN. That was a real
bug on the first run.

## dune — anisotropic ridges with a slip face

`sin(φ + a·sin φ)` along a wind direction. The skew `a` is the windward/lee
asymmetry of a migrating dune: gentle windward flank, steep slip face.

- Mapped to [0, 1] — dunes add relief, they never dig.
- Sharpened by an exponent to narrow the crests.
- Phase-warped by low-frequency noise so crest lines meander instead of running
  dead straight across the map.

## terrace — stepped benches

Quantise the base into `steps` benches with a soft step

```
t_s = t^p / (t^p + (1−t)^p),      p = 1 / (1 − sharpness)
```

and return `quantised − base`. `sharpness = 0` gives `p = 1` and the identity;
`sharpness → 1` gives a hard riser. Step edges are noise-warped so the contours
are irregular.

**Steps are counted per normalised height unit**, so bench height in metres is
`height_scale_m / steps`. This bit users: a plateau with 9 steps over 320 m of
relief has 35 m benches, and noise of ±16 m then never crosses a step — the
plateau comes out dead flat. The planner derives the step count from scene
relief for exactly this reason, targeting ~12 m benches at any scale.

## erosion — thermal relaxation to the angle of repose

Material above the talus gradient slides to lower neighbours, distributed
proportionally over the four neighbours.

- The talus threshold is a real gradient: the repose angle converted through the
  cell size, so the same spec erodes consistently at any raster resolution.
- Unconditionally stable for `strength ≤ 1`.
- **Mass-conserving to machine precision**, and tested as such. Getting there
  required distinguishing two shift semantics: edge clamping when measuring
  height differences, so no artificial slope appears at the border and nothing
  flows off the map; zero fill when redistributing, so nothing arrives from
  outside. Clamping both makes a border row quietly manufacture material every
  iteration.

Erosion is what produces talus fans and rounded scree that pure noise never
gives, and it is the operator most responsible for a terrain reading as
*eroded ground* rather than *filtered noise*.

## Noise, since the operators sit on top of it

Gradient (Perlin) noise, not value noise. Value noise puts its extrema **on** the
lattice points, and terracing turns that into a visible rectangular maze — which
is precisely what the first canyon render showed. Gradient noise puts zeros on
the lattice instead, which reads as organic.

Each octave is additionally rotated by the golden angle, because even gradient
noise leaks its axes when octaves stack in register.

Frequencies are in cycles across the world extent, so a spec produces the same
landform at any raster resolution — only the sampling gets finer.
