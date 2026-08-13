"""Stage 1: intent analysis, then planning.

The paper keeps these strictly apart and so do we.  Intent analysis **extracts
only what the prompt says** and leaves everything else empty; planning then
fills the gaps and records which fields it invented.  Merging the two would let
a confident model overwrite what the user actually asked for -- and there would
be no way afterwards to tell the difference.
"""

from __future__ import annotations

from ..schemas import SceneIntent
from .schema import ScenePlan

INTENT_SYSTEM = """\
You extract scene requirements from a user's prompt for a 3D landscape generator.

Extract ONLY what the prompt actually states or unambiguously implies by its own
words. This is the critical rule: if the prompt does not mention something, leave
that field empty or null. Do NOT invent, complete, embellish or infer a "typical"
value. An empty field is a correct answer; a plausible guess is a wrong one,
because a later planning step needs to know what the user did not specify.

Field guidance:
- biome: only if a landscape type is named or plainly described.
- landforms: named terrain features (canyon, dunes, cliff, lake, plateau).
- objects: things placed in the world (huts, trees, rocks, ruins).
- mood / time_of_day: only if stated.
- scale_hint: only if size or distance is mentioned.
- explicit_constraints: instructions the user gave that must be honoured
  verbatim, e.g. "no water", "must be walkable".
"""

PLAN_SYSTEM = """\
You are the planning stage of a 3D landscape generator. You receive the user's
prompt and a strict extraction of what it stated. Produce a complete scene plan.

Your job is to fill the gaps the extraction left open, and to do so coherently.
Never contradict the extracted intent: anything the user stated is fixed, and
you design around it.

Rules:
- 2 to 5 regions. They must partition the landscape sensibly, not overlap
  conceptually. Use the given categories only.
- elevation is relative within this scene: the lowest region should be near 0.0
  and the highest near 1.0, so the scene actually uses its relief.
- relative_area values should roughly sum to 1.
- character describes landform shape, not parameters. Pick at most 3 words.
- category is what the ground IS, and it must fit the biome. 'plain' means
  GRASSLAND and renders green; an arid scene wants sand, dune, slope, plateau
  or rock instead. Picking 'plain' for a desert floor produces a green desert.
- color_hint is what the ground LOOKS LIKE. Set it on every region, reasoning
  from the biome, not only when the prompt names a colour: a dried-out canyon
  rim is ochre or brown, its walls red, its floor grey -- none of that is
  stated outright, and all of it is obvious from "dried-out red sandstone
  canyon". This is the only path from the scene's look to the render; a null
  hint falls back to the category's own colour, which assumes a temperate
  biome. Leave it null only when that fallback is already correct.
- relief_m is the height difference between the lowest and highest ground.
  A dune field is 40-120 m. A canyon or mountain scene is 300-800 m.
- scatter lists small props strewn over a region, never buildings.
- layout.image_prompt describes a TOP-DOWN CATEGORICAL MAP, not a landscape
  painting. Ask explicitly for large flat areas of solid colour with hard
  edges, no shading, no gradients, no texture, no text, no perspective.
- inferred_fields: list every field you chose that the prompt did not state.
  Be honest and thorough here; it is used to show the user what was assumed.
"""


def build_plan_user_message(prompt: str, intent: SceneIntent) -> str:
    return (
        f"User prompt:\n{prompt}\n\n"
        f"Strictly extracted intent (empty fields mean the user said nothing "
        f"about them -- you decide those):\n{intent.model_dump_json(indent=2)}"
    )


def analyse_intent(llm, prompt: str) -> SceneIntent:
    """Extraction only. Never fills gaps."""
    intent = llm.complete_json(INTENT_SYSTEM, f"Prompt:\n{prompt}", SceneIntent, "SceneIntent")
    # The model does not get to rewrite the prompt it was given.
    return intent.model_copy(update={"prompt": prompt})


def plan_scene(llm, prompt: str, intent: SceneIntent) -> ScenePlan:
    """Gap filling. Must not contradict the extracted intent."""
    return llm.complete_json(
        PLAN_SYSTEM, build_plan_user_message(prompt, intent), ScenePlan, "ScenePlan"
    )
