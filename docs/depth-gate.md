# Topography fidelity gate

The sharpest metric in the acceptance criteria, and the guard for the project's
largest risk.

## Why it exists

Stage 3 renders the terrain, has an image model insert objects into that render,
then back-projects rays through the **original** camera to place the results.
The whole construction assumes the edit left the terrain and the viewpoint
alone.

If the model instead nudges a ridge, shifts the horizon or subtly re-frames the
shot, every ray lands somewhere else. The placements are wrong, and nothing
downstream can tell: the edited image still looks fine, the meshes still
reconstruct, the objects still land *somewhere*. This is exactly the failure
mode the briefing's risk table puts first.

So the edit is checked before it is used, and rejected if it moved the ground.

## The asymmetry that makes it work

**An inserted object can only bring the surface closer to the camera.** It
occludes terrain; it cannot reveal anything behind it.

So depth *decreasing* is expected and allowed, while depth *increasing* means
the terrain itself moved. That one-sided test separates legitimate edits from
broken ones without needing to know where the objects were placed — which is
useful, because at gate time we do not yet know.

Two bounds are applied:

- **`raised_fraction`** — share of pixels that moved *away* from the camera.
  Default limit 2%. This is the real test.
- **`changed_fraction`** — share that changed at all. Default limit 45%. An edit
  that covers most of the frame is unusable to back-project through even if
  every change is strictly nearer.

## Working with estimated depth

The pre-edit image can in principle get a true depth pass from Blender, but the
edited image exists only as pixels, so its depth must be estimated — and
monocular estimators are ambiguous up to scale and offset. Comparing raw values
would flag every edit.

`align_depth` fits `after ≈ a * before + b` and judges the residual. The fit is
**trimmed**: the pixels the edit legitimately changed are exactly the outliers,
so including them would drag the alignment towards hiding the very thing being
tested. `test_alignment_does_not_hide_a_moved_terrain` pins that.

**Supported path: run the same estimator on both images.** Its systematic bias
then cancels, which is what the briefing describes. The Blender Z-pass is a
bonus for validating the estimator, not a requirement — and at present Blender
5's compositor file output is in flux, so `--depth` warns loudly when it
produces nothing rather than leaving the gate comparing an estimate against
nothing.

## Known limitation

A camera dolly along its own optical axis shifts every depth by a constant — and
that is exactly what the alignment step is built to absorb. **This gate cannot
see it.**

It covers terrain deformation and any re-framing that changes depth *structure*:
panning, tilting, a moved horizon, a flattened ridge. A pure dolly has to be
prevented on the prompt side, by never asking the editor to change the
viewpoint.

`test_a_pure_depth_offset_is_knowingly_undetectable` asserts the limitation, so
it stays a documented gap rather than becoming a surprise later.

## Tested cases

| Edit | Verdict |
|---|---|
| Nothing changed | accept (with a note: did the edit insert anything?) |
| One or several objects inserted | accept |
| Estimator noise | accept |
| Arbitrary scale and offset on the estimate | accept, alignment absorbs it |
| A ridge pushed away from the camera | **reject** — terrain moved |
| Lateral pan, tilted horizon | **reject** |
| Wholesale repaint | **reject** |
| Object covering the whole frame | **reject** — too invasive |
| Pure depth offset (dolly) | accept — documented limitation |

The report carries numbers, not just a verdict, because a rejected edit gets
retried and the retry needs to be diagnosable.
