"""Tests for the planning stage (M3).

No model runs here.  A scripted LLM replays prepared objects, which is what
makes prompts, schema handling, the intent/planning split and the layout gate
testable at all -- the real endpoint lives on the user's machine and needs a GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from worldclaw.planning import agent
from worldclaw.planning.expand import expand
from worldclaw.planning.layout_image import (
    clean_layout,
    evaluate_layout,
    palette_prompt,
)
from worldclaw.planning.llm import LLMError, ScriptedLLM, inline_refs
from worldclaw.planning.schema import LayoutPlan, RegionPlan, ScenePlan
from worldclaw.schemas import SceneIntent, TerrainSpec


@pytest.fixture
def plan() -> ScenePlan:
    return ScenePlan(
        name="red-gorge",
        biome="arid canyon",
        world_size_m=2048.0,
        relief_m=420.0,
        regions=[
            RegionPlan(name="gorge floor", category="basin", relative_area=0.15,
                       elevation=0.04, character=["eroded"], scatter=["boulder", "shrub"]),
            RegionPlan(name="talus slope", category="slope", relative_area=0.25,
                       elevation=0.45, character=["rugged", "eroded"]),
            RegionPlan(name="upper plateau", category="plateau", relative_area=0.5,
                       elevation=0.88, character=["terraced"]),
            RegionPlan(name="massif", category="rock", relative_area=0.1,
                       elevation=1.0, character=["peaked", "rugged"]),
        ],
        layout=LayoutPlan(composition="channel", image_prompt="a canyon cutting north to south"),
        inferred_fields=["relief_m", "world_size_m"],
    )


# ------------------------------------------------------------------ schema


def test_inline_refs_removes_all_references():
    """Schema-constrained decoding must not depend on the server following $ref."""
    schema = inline_refs(ScenePlan.model_json_schema())
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    # The nested structure must survive inlining, not be flattened away.
    assert schema["properties"]["regions"]["items"]["properties"]["category"]["enum"]
    assert schema["properties"]["layout"]["properties"]["composition"]["enum"]


def test_inline_refs_survives_recursive_models():
    from pydantic import BaseModel

    class Node(BaseModel):
        name: str
        child: "Node | None" = None

    Node.model_rebuild()
    schema = inline_refs(Node.model_json_schema())
    assert "$defs" not in json.dumps(schema)


# ------------------------------------------------------- intent / planning


def test_intent_analysis_keeps_the_original_prompt():
    """The model does not get to restate the user's prompt."""
    llm = ScriptedLLM([SceneIntent(prompt="something else", biome="desert")])
    intent = agent.analyse_intent(llm, "A windswept dune sea")
    assert intent.prompt == "A windswept dune sea"
    assert intent.biome == "desert"


def test_planning_receives_the_extracted_intent(plan):
    """The planner must see which fields the extraction deliberately left empty."""
    intent = SceneIntent(prompt="A canyon", landforms=["canyon"])
    llm = ScriptedLLM([plan])
    agent.plan_scene(llm, "A canyon", intent)
    system, user = llm.calls[0]
    assert "canyon" in user
    assert '"biome": null' in user  # the gap is visible to the planner
    assert "you decide those" in user


def test_intent_prompt_forbids_invention():
    assert "ONLY" in agent.INTENT_SYSTEM
    assert "empty field is a correct answer" in agent.INTENT_SYSTEM


def test_scripted_llm_reports_a_missing_response():
    with pytest.raises(LLMError, match="ScenePlan"):
        agent.plan_scene(ScriptedLLM([]), "x", SceneIntent(prompt="x"))


# ---------------------------------------------------------------- expansion


def test_expansion_produces_a_valid_terrain_spec(plan):
    spec = expand(plan, seed=42, resolution=256)
    assert isinstance(spec, TerrainSpec)
    assert spec.height_scale_m == pytest.approx(420.0)
    assert spec.resolution == 256 and spec.seed == 42
    assert [r.name for r in spec.regions] == [
        "gorge_floor", "talus_slope", "upper_plateau", "massif"
    ]


def test_expansion_maps_character_to_operators(plan):
    spec = expand(plan)
    ops = {r.name: [o.kind for o in r.operators] for r in spec.regions}
    assert "terrace" in ops["upper_plateau"]
    assert "peak" in ops["massif"]
    assert all("erosion" in v for v in ops.values())
    # "eroded" must actually erode harder than the default.
    floor = next(r for r in spec.regions if r.name == "gorge_floor")
    erosion = next(o for o in floor.operators if o.kind == "erosion")
    assert erosion.iterations >= 50


def test_terrace_step_height_tracks_the_scene_relief():
    """Benches must stay a sensible height whatever the scene's scale.

    Steps are counted per normalised unit, so a fixed step count gives 3 m
    benches in a dune field and 60 m benches in a mountain range.
    """
    heights = []
    for relief in (80.0, 400.0, 1200.0):
        p = ScenePlan(
            name="t", biome="b", world_size_m=1024.0, relief_m=relief,
            regions=[
                RegionPlan(name="low", category="basin", relative_area=0.5, elevation=0.0),
                RegionPlan(name="high", category="plateau", relative_area=0.5,
                           elevation=1.0, character=["terraced"]),
            ],
            layout=LayoutPlan(composition="bands", image_prompt="x"),
        )
        spec = expand(p)
        high = next(r for r in spec.regions if r.name == "high")
        terrace = next(o for o in high.operators if o.kind == "terrace")
        heights.append(spec.height_scale_m / terrace.steps)
    assert all(4.0 <= h <= 25.0 for h in heights), heights


def test_expansion_separates_regions_sharing_a_category():
    """Two regions of one category must not collide on the layout map."""
    p = ScenePlan(
        name="t", biome="b", world_size_m=1024.0, relief_m=100.0,
        regions=[
            RegionPlan(name="north rocks", category="rock", relative_area=0.5, elevation=0.9),
            RegionPlan(name="south rocks", category="rock", relative_area=0.5, elevation=0.4),
        ],
        layout=LayoutPlan(composition="patches", image_prompt="x"),
    )
    spec = expand(p)  # TerrainSpec itself rejects duplicate colours
    assert spec.regions[0].color != spec.regions[1].color


def test_unknown_scatter_class_is_kept_not_dropped():
    p = ScenePlan(
        name="t", biome="b", world_size_m=1024.0, relief_m=100.0,
        regions=[
            RegionPlan(name="a", category="plain", relative_area=0.5, elevation=0.2,
                       scatter=["mushroom"]),
            RegionPlan(name="b", category="rock", relative_area=0.5, elevation=0.8),
        ],
        layout=LayoutPlan(composition="bands", image_prompt="x"),
    )
    spec = expand(p)
    assert [s.asset_class for s in spec.regions[0].scatter] == ["mushroom"]


# -------------------------------------------------------------- layout gate


def _render_spec_map(spec: TerrainSpec, res: int = 128) -> np.ndarray:
    """A clean two-band map in the spec's exact colours."""
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    step = res // len(spec.regions)
    for i, r in enumerate(spec.regions):
        rgb[i * step : (i + 1) * step or res] = r.rgb
    rgb[len(spec.regions) * step :] = spec.regions[-1].rgb
    return rgb


def test_palette_prompt_names_every_colour(plan):
    spec = expand(plan)
    text = palette_prompt(spec, "a canyon")
    for r in spec.regions:
        assert r.color in text
    assert "no gradients" in text and "no text" in text


def test_gate_accepts_an_exact_palette_map(plan):
    spec = expand(plan)
    report = evaluate_layout(_render_spec_map(spec), spec)
    assert report.accepted and report.unmatched_fraction == pytest.approx(0.0)


def test_gate_tolerates_noise_and_soft_edges(plan):
    """Anti-aliasing and mild noise are expected from any image model."""
    spec = expand(plan)
    rgb = _render_spec_map(spec).astype(np.int16)
    gen = np.random.default_rng(0)
    rgb += gen.integers(-12, 13, rgb.shape)
    report = evaluate_layout(np.clip(rgb, 0, 255).astype(np.uint8), spec)
    assert report.accepted, report.reason


def test_gate_rejects_a_map_that_ignored_the_palette(plan):
    """A model that painted a pretty landscape instead of a category map."""
    spec = expand(plan)
    gen = np.random.default_rng(1)
    rgb = gen.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    report = evaluate_layout(rgb, spec)
    assert not report.accepted
    assert "palette mismatch" in report.reason


def test_gate_rejects_a_map_missing_a_region(plan):
    spec = expand(plan)
    rgb = np.full((128, 128, 3), spec.regions[0].rgb, dtype=np.uint8)
    report = evaluate_layout(rgb, spec)
    assert not report.accepted
    assert report.missing_regions


def test_clean_layout_snaps_to_the_palette_and_despeckles(plan):
    spec = expand(plan)
    clean = _render_spec_map(spec)
    noisy = clean.copy()
    gen = np.random.default_rng(2)
    ys, xs = gen.integers(0, 128, (2, 400))
    noisy[ys, xs] = gen.integers(0, 256, (400, 3), dtype=np.uint8)

    out = clean_layout(noisy, spec)
    allowed = {r.rgb for r in spec.regions}
    assert {tuple(c) for c in np.unique(out.reshape(-1, 3), axis=0)} <= allowed
    # The speckle must be gone, not merely quantised into stray regions.
    assert float((out != clean).any(-1).mean()) < 0.02


def test_procedural_backend_matches_planned_region_colours(plan, tmp_path):
    """The template painter must work for planner-named regions, not just presets."""
    from worldclaw.planning.layout_image import generate_layout

    spec = expand(plan, resolution=192)
    rgb, report = generate_layout(
        spec, "x", tmp_path / "layout.png", backend="procedural", template="canyon"
    )
    used = {tuple(c) for c in np.unique(rgb.reshape(-1, 3), axis=0)}
    assert used <= {r.rgb for r in spec.regions}
    assert report.unmatched_fraction == pytest.approx(0.0)


# ------------------------------------------------------------- colour hints


def test_colour_hint_tints_the_material_and_keeps_category_brightness():
    """A stated colour must actually reach the render.

    Without this the material comes from the category table alone, so "red
    sandstone" arrives as category 'rock' and renders neutral grey -- which is
    what made a real user's whole canyon look like snow while every brightness
    diagnostic reported healthy.
    """
    from worldclaw.planning.expand import _MATERIAL, _apply_color_hint, _luma

    base = _MATERIAL["rock"][0]
    red = _apply_color_hint(base, "red")

    assert red[0] > red[1] and red[0] > red[2]          # actually red
    assert (max(red) - min(red)) / max(red) > 0.4       # actually saturated
    assert _luma(red) == pytest.approx(_luma(base), rel=0.02)  # rock stays rock-dark


def test_achromatic_hints_change_lightness_rather_than_hue():
    """'chalk cliffs' means pale and 'basalt' means dark.

    Renormalising these back to the category's luminance -- correct for a hue
    hint -- would cancel exactly what was asked for.
    """
    from worldclaw.planning.expand import _MATERIAL, _apply_color_hint, _luma

    base = _MATERIAL["rock"][0]
    assert _luma(_apply_color_hint(base, "white")) > _luma(base) * 1.5
    assert _luma(_apply_color_hint(base, "black")) < _luma(base) * 0.6


@pytest.mark.parametrize("category", ["rock", "sand", "dune", "plain", "peak"])
@pytest.mark.parametrize("hint", ["red", "orange", "ochre", "green", "blue", "white", "black"])
def test_colour_hints_never_clip_a_channel(category, hint):
    """A clipped channel bends the hue away from the colour that was asked for."""
    from worldclaw.planning.expand import _MATERIAL, _apply_color_hint

    out = _apply_color_hint(_MATERIAL[category][0], hint)
    assert all(0.0 <= c <= 1.0 for c in out), out


def test_no_hint_leaves_the_category_colour_untouched():
    from worldclaw.planning.expand import _MATERIAL, _apply_color_hint

    base = _MATERIAL["sand"][0]
    assert _apply_color_hint(base, None) == pytest.approx(base)
    assert _apply_color_hint(base, "chartreuse") == pytest.approx(base)  # unknown word


def test_colour_hint_survives_the_full_expansion(plan):
    """The hint has to arrive in the TerrainSpec's material, not just in the plan."""
    plan.regions[3].color_hint = "red"          # the massif
    spec = expand(plan)
    massif = next(r for r in spec.regions if r.name == "massif")
    c = massif.material.base_color
    assert c[0] > c[1] and c[0] > c[2]


def test_two_regions_of_one_category_can_differ_by_colour():
    """The layout-map colours were already separated for duplicate categories;
    the *materials* were not, so both rendered identically."""
    p = ScenePlan(
        name="t", biome="b", world_size_m=1024.0, relief_m=200.0,
        regions=[
            RegionPlan(name="red walls", category="rock", relative_area=0.5,
                       elevation=0.8, color_hint="red"),
            RegionPlan(name="grey scree", category="rock", relative_area=0.5,
                       elevation=0.2, color_hint="grey"),
        ],
        layout=LayoutPlan(composition="bands", image_prompt="x"),
    )
    spec = expand(p)
    a, b = (r.material.base_color for r in spec.regions)
    assert a != pytest.approx(b)
