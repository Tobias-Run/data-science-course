"""Pre-flight check: is this machine ready for a full run?

Every failure mode below has a specific, actionable fix, and each one otherwise
surfaces much later as a confusing error in the middle of a stage.  Running this
first turns "it crashed" into "install X" before anything expensive starts.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass
class Probe:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str = ""


def _python() -> Probe:
    major, minor = sys.version_info[:2]
    v = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) == (3, 11):
        return Probe("python", "ok", f"{v} — the version bpy ships wheels for")
    return Probe(
        "python", "warn", f"{v}",
        "bpy publishes wheels for CPython 3.11 only. On another version the "
        "terrain stage still runs; only the Blender stage is unavailable. "
        "Fix with a 3.11 environment: python3.11 -m venv .venv && "
        "source .venv/bin/activate && pip install -e '.[blender,dev]'",
    )


def _core() -> list[Probe]:
    out = []
    for mod, why in (("numpy", "arrays"), ("scipy", "filters and distance transforms"),
                     ("PIL", "image I/O"), ("pydantic", "the artefact schemas")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", getattr(m, "VERSION", "?"))
            out.append(Probe(mod, "ok", f"{ver} — {why}"))
        except ImportError:
            out.append(Probe(mod, "fail", "missing", "pip install -e '.[dev]'"))
    return out


def _blender() -> Probe:
    try:
        import bpy

        return Probe("bpy", "ok", f"Blender {bpy.app.version_string} as a python module")
    except ImportError:
        return Probe(
            "bpy", "warn", "not importable",
            "pip install 'bpy>=5.0' (needs CPython 3.11). Without it the terrain "
            "stage still produces the height field, OBJ and 16-bit map; only the "
            "renders, glTF and .blend are skipped.",
        )


def _llm(base_url: str, model: str) -> list[Probe]:
    from .planning.llm import LLMConfig, LLMError, LocalLLM

    llm = LocalLLM(LLMConfig(base_url=base_url, model=model))
    try:
        models = llm.available_models()
    except LLMError:
        return [Probe(
            "llm endpoint", "fail", f"{base_url} unreachable",
            "In LM Studio: load a model, open the Developer tab and press Start "
            "Server. The default address is http://localhost:1234/v1. Other "
            "servers work too — pass --llm-url.",
        )]

    probes = [Probe("llm endpoint", "ok", f"{base_url} serves {len(models)} model(s)")]
    if model in models:
        probes.append(Probe("llm model", "ok", model))
    elif models:
        probes.append(Probe(
            "llm model", "warn", f"{model!r} is not in the served list",
            f"pass --llm-model {models[0]!r}, or load that model in LM Studio",
        ))
    else:
        probes.append(Probe("llm model", "fail", "the endpoint serves nothing",
                            "load a model in LM Studio before starting the server"))
    return probes


def _structured_output(base_url: str, model: str) -> Probe:
    """The planning stage is unusable without schema-constrained decoding."""
    from pydantic import BaseModel

    from .planning.llm import LLMConfig, LLMError, LocalLLM

    class Ping(BaseModel):
        biome: str
        regions: int

    try:
        got = LocalLLM(LLMConfig(base_url=base_url, model=model, max_tokens=200)).complete_json(
            "You answer with JSON only.",
            "Describe a desert scene: biome name and a number of regions.",
            Ping,
            "Ping",
        )
        return Probe("structured output", "ok",
                     f"schema honoured (biome={got.biome!r}, regions={got.regions})")
    except LLMError as exc:
        return Probe(
            "structured output", "fail", str(exc).splitlines()[0][:160],
            "The planner needs the endpoint to enforce a JSON schema. Update LM "
            "Studio, and prefer an instruct model that advertises structured "
            "output / function calling. Very small models often fail this.",
        )


def _imagegen() -> list[Probe]:
    out = []
    try:
        import torch

        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
               else "cpu")
        out.append(Probe("torch", "ok", f"{torch.__version__}, device {dev}"))
        if dev == "cpu":
            out.append(Probe("gpu", "warn", "no GPU visible to torch",
                             "A layout map on CPU is slow but works. Use "
                             "--layout-backend procedural to skip it entirely."))
    except ImportError:
        out.append(Probe("torch", "warn", "not installed",
                         "Optional. pip install 'worldclaw[imagegen]' for the "
                         "diffusers layout backend; the procedural backend needs nothing."))
    try:
        import diffusers

        out.append(Probe("diffusers", "ok", diffusers.__version__))
    except ImportError:
        out.append(Probe("diffusers", "warn", "not installed",
                         "Optional, as above."))
    return out


def run(base_url: str, model: str, check_llm: bool = True) -> list[Probe]:
    probes = [_python(), *_core(), _blender()]
    if check_llm:
        llm_probes = _llm(base_url, model)
        probes += llm_probes
        if all(p.status != "fail" for p in llm_probes):
            probes.append(_structured_output(base_url, model))
    probes += _imagegen()
    return probes


def format_report(probes: list[Probe]) -> str:
    mark = {"ok": " ok ", "warn": "warn", "fail": "FAIL"}
    lines = [f"worldclaw doctor — {platform.platform()}", ""]
    for p in probes:
        lines.append(f"  [{mark[p.status]}] {p.name:<18} {p.detail}")
        if p.fix:
            for i, chunk in enumerate(_wrap(p.fix, 66)):
                lines.append(f"           {'→ ' if i == 0 else '  '}{chunk}")
    fails = [p for p in probes if p.status == "fail"]
    warns = [p for p in probes if p.status == "warn"]
    lines += ["", f"  {len(probes) - len(fails) - len(warns)} ok, "
                  f"{len(warns)} warnings, {len(fails)} blocking"]
    if not fails:
        lines.append("  ready for a full run")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
