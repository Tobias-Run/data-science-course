# Ray-pair placement (paper section 3.2)

How a reconstructed object mesh becomes a correctly sized, correctly located,
ground-touching object in the world. This is the most delicate arithmetic in the
project, so it is written down in full.

## The problem

Stage 3 renders the terrain, has an image model insert objects into that render,
segments them, and reconstructs a mesh per object. What comes back is a mesh in
*some* camera's frame at *some* scale. Nothing about it says where in the world
the object belongs or how big it is.

Everything needed to answer that is already known: the render's camera, the
terrain it was rendered from, and the pixel the object occupies.

## The chain

### 1. Crop bookkeeping

The object is cut out of the composition image and enlarged before
reconstruction. With `A_i` the affine from composition pixels to object-centred
crop pixels:

```
K_hat_i = A_i @ K_t          extrinsics unchanged: E_t
```

Cropping and scaling move the *image coordinate system*, not the camera. This is
the identity that lets reconstruction run at high resolution without costing
placement accuracy, so it is tested directly
(`test_equivalent_intrinsics_match_an_actually_cropped_projection`) rather than
assumed.

### 2. Scale calibration

Single-image reconstruction drifts in scale. A factor `lambda_i` about the mesh
centre is fitted until the projected bounding-box area matches the reference
mask's, within **asymmetric** tolerance `[-eps_under, +eps_over]`.

Asymmetric because the two errors are not equally bad: an object that is too
large reads as broken, one slightly too small reads as further away. The default
band is 10% under, 5% over.

Scaling about the centroid moves the mesh in depth as well as size, so projected
area is not exactly quadratic in the factor — hence a bisection on the log
factor rather than a closed form.

### 3. Two rays

**First ray** — through the object's centre, into the reconstructed mesh, in the
**object camera's** frame. Gives the reference point `P_o` and its depth `Z_o`.

> The intrinsics here are the object camera's (`f_o`, principal point at the
> crop centre), **not** `K_hat_i`. The mesh lives in the object camera's frame;
> a ray built from `K_hat_i` is expressed in the terrain camera's frame and
> shooting it into that mesh is plausible-looking and wrong. This was a real bug,
> caught by the round-trip test.

**Second ray** — the crop centre mapped back through `A_i^-1` to a composition
pixel, then cast from the terrain camera and intersected with the height field.
Gives the anchor `P_t` and its depth `Z_t`.

The terrain intersection marches coarsely to bracket the first sign change of
`ray.z - H(ray.xy)`, then bisects. Marching alone would quantise the anchor to
the step length, and the anchor's depth divides directly into the scale — so the
error would surface as objects of the wrong *size*, not the wrong place.

### 4. Similarity transform

```
s_i = (Z_t / Z_o) * (f_o / f_hat_i)

            [ s_i * R_i    P_t - s_i * R_i * P_o ]
T_place  =  [                                    ]  @  T_l2c
            [ 0^T          1                     ]
```

`s_i` makes the object subtend the same angle from the terrain camera that it
did in its own reconstruction. `R_i = R_t @ R_o^T` carries the object's world
orientation across. The translation puts `P_o` exactly on `P_t`.

The result is in terrain-camera coordinates; `E_t^-1` takes it to world.

### The double-transform trap

`T_place` ends in `@ T_l2c`. **If the exporter already baked the local-to-camera
transform into the vertices, applying it again silently doubles the rotation and
translation.** `PlacementRecord.l2c_baked_into_vertices` records which case an
asset is, and `compose_transform` raises rather than guessing when it is told the
transform is not baked but is given no matrix.

### 5. Contact search

Against residual floating: anchor depth and scale are varied **together** along
the terrain ray. Moving the anchor to depth `k * Z_t` and scaling by the same `k`
leaves the projection identical — the object keeps looking exactly as composed
while its contact with the ground changes. That invariance is what makes the
search free of visual cost, and it is the property to protect if the search is
ever extended (`test_contact_search_preserves_the_projection`).

Contact is measured as in M2: the share of the object's bottom band within
tolerance of the terrain, plus the largest gap and the largest penetration. One
definition across milestones, so the numbers are comparable.

## Frame conventions

All camera maths is **computer-vision convention**: x right, y down, z forward,

```
u = fx * P.x / P.z + cx        v = fy * P.y / P.z + cy
```

with image row 0 at the top. Blender's camera looks down its local −z with +y
up, which is a *different* convention. The Blender stage therefore writes
`E_world_to_cam_cv` alongside the raw matrix and this package consumes only that
one. Mixing them mirrors the scene vertically, and the result looks like a
plausible-but-wrong placement rather than an error.

World space is the terrain frame: x east, y north, z up, origin at the centre of
the world.

## How it is tested without any model

The load-bearing test is a **round trip**: build a scene where the true
placement is known, compute what the cameras would have seen, run the placement,
and check it recovers what it started from.

One subtlety makes or breaks that fixture. A single-image reconstruction is
**not** metrically true — it comes back at whatever scale reproduces the crop's
appearance from its own camera. That is precisely why `s = (Z_t/Z_o)(f_o/f_hat)`
exists. Building the fixture with a metrically correct mesh instead makes the
placement look 3.5x wrong when it is doing exactly its job. (It did, at first.)

So the synthetic reconstruction is scaled by the factor that makes its apparent
size match the crop's, and the reconstruction camera views the object from the
same direction the terrain camera does — as SAM3D would, since it reconstructs
from that very crop.

Covered:

| Property | Test |
|---|---|
| Anchor lands on the true ground point, flat and sloped | `test_round_trip_recovers_the_anchor_*` |
| Placed object has its true metric size | `test_round_trip_recovers_the_metric_object_size` |
| Apparent size preserved regardless of metric scale | `test_placed_object_keeps_the_apparent_size...` |
| Placement independent of crop resolution | `test_round_trip_is_independent_of_crop_resolution` |
| `s` proportional to `f_o`, inversely to crop enlargement | `test_scale_tracks_...`, `test_scale_is_inversely_...` |
| Double transform is refused, not guessed | `test_compose_refuses_to_guess...` |
| Contact search preserves the projection | `test_contact_search_preserves_the_projection` |
