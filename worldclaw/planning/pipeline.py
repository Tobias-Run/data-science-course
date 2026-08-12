"""Stage 1 driver: prompt in, TerrainSpec and layout map out.

Cached like every other stage, keyed on the prompt and the model settings, so a
re-run does not spend another model call to get the same plan back.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import RunContext
from ..schemas import SceneIntent, SceneSpec, TerrainSpec
from . import agent
from .expand import expand
from .layout_image import generate_layout, palette_prompt
from .schema import ScenePlan


def run_planning_stage(
    llm,
    prompt: str,
    run: RunContext,
    seed: int = 0,
    resolution: int = 768,
    layout_backend: str = "procedural",
    layout_template: str = "canyon",
    model_id: str = "stabilityai/sdxl-turbo",
    llm_label: str = "local",
) -> tuple[SceneSpec, Path]:
    intent_json = run.path("plan", "intent.json")
    plan_json = run.path("plan", "scene_plan.json")
    spec_json = run.path("spec", "terrain_spec.json")

    key = {
        "prompt": prompt,
        "seed": seed,
        "resolution": resolution,
        "llm": llm_label,
    }

    with run.stage("plan", key, [intent_json, plan_json, spec_json]) as todo:
        if todo:
            intent = agent.analyse_intent(llm, prompt)
            run.write_model(intent, "plan", "intent.json")
            plan = agent.plan_scene(llm, prompt, intent)
            run.write_model(plan, "plan", "scene_plan.json")
            run.write_model(expand(plan, seed=seed, resolution=resolution),
                            "spec", "terrain_spec.json")

    intent = SceneIntent.model_validate_json(intent_json.read_text())
    plan = ScenePlan.model_validate_json(plan_json.read_text())
    spec = TerrainSpec.model_validate_json(spec_json.read_text())

    # ---- layout map --------------------------------------------------------
    layout_path = run.path("plan", "layout.png")
    report_json = run.path("plan", "layout_report.json")
    layout_key = {**key, "backend": layout_backend, "template": layout_template,
                  "model": model_id, "spec": spec.model_dump(mode="json")}

    with run.stage("layout_map", layout_key, [layout_path, report_json]) as todo:
        if todo:
            _, report = generate_layout(
                spec,
                palette_prompt(spec, plan.layout.image_prompt),
                layout_path,
                backend=layout_backend,
                template=layout_template,
                model_id=model_id,
            )
            run.write_json(report.model_dump(), "plan", "layout_report.json")

    spec = spec.model_copy(update={"layout_map": str(layout_path)})
    run.write_model(spec, "spec", "terrain_spec.json")

    scene = SceneSpec(
        name=spec.name,
        intent=intent,
        terrain=spec,
        inferred_fields=plan.inferred_fields,
    )
    run.write_model(scene, "scene_spec.json")
    return scene, layout_path
