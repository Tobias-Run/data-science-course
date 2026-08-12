"""Run directory, artefact I/O and stage-level resumability.

Every stage writes files and a manifest entry.  Re-running a pipeline skips a
stage whose input fingerprint and output files are both unchanged, so an
expensive stage (a model call, an erosion sweep) is paid for once per run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

_ARTIFACT_ROOT = Path("artifacts")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(obj: Any) -> str:
    """Stable content hash of any JSON-able object (pydantic models included)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


@dataclass
class StageResult:
    name: str
    status: str  # "ran" | "cached"
    seconds: float
    outputs: list[str] = field(default_factory=list)


class RunContext:
    """A single generation run rooted at ``artifacts/<run_id>``."""

    def __init__(self, run_id: str, root: Path | str = _ARTIFACT_ROOT, force: bool = False):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.manifest_path = self.dir / "manifest.json"
        self.manifest: dict[str, Any] = self._load_manifest()
        self.results: list[StageResult] = []

    # ---------------------------------------------------------------- paths
    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, path: str | Path) -> str:
        path = Path(path)
        try:
            return str(path.relative_to(self.dir))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------- manifest
    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {
            "run_id": self.run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(),
            "stages": {},
        }

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")

    # ---------------------------------------------------------------- stages
    @contextmanager
    def stage(self, name: str, inputs: Any, outputs: list[str | Path]) -> Iterator[bool]:
        """Run a stage unless a cached result with the same inputs is on disk.

        Yields ``True`` when the body should do the work, ``False`` when the
        cached artefacts are still valid::

            with run.stage("heightfield", spec, [out]) as todo:
                if todo:
                    ...expensive work...
        """
        want = fingerprint(inputs)
        rec = self.manifest["stages"].get(name)
        have_files = all(Path(o).exists() for o in outputs)
        cached = bool(rec) and rec.get("inputs") == want and have_files and not self.force

        if cached:
            self.results.append(StageResult(name, "cached", 0.0, [self.rel(o) for o in outputs]))
            yield False
            return

        t0 = time.perf_counter()
        yield True
        dt = time.perf_counter() - t0

        missing = [str(o) for o in outputs if not Path(o).exists()]
        if missing:
            raise RuntimeError(f"stage {name!r} did not produce: {missing}")

        self.manifest["stages"][name] = {
            "inputs": want,
            "outputs": [self.rel(o) for o in outputs],
            "seconds": round(dt, 3),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save_manifest()
        self.results.append(StageResult(name, "ran", dt, [self.rel(o) for o in outputs]))

    # ------------------------------------------------------------------- io
    def write_model(self, model: BaseModel, *parts: str) -> Path:
        p = self.path(*parts)
        p.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return p

    def write_json(self, obj: Any, *parts: str) -> Path:
        p = self.path(*parts)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return p

    def report(self) -> str:
        lines = [f"run {self.run_id}  ({self.dir})"]
        for r in self.results:
            mark = "cache" if r.status == "cached" else f"{r.seconds:6.2f}s"
            lines.append(f"  [{mark:>7}] {r.name}")
        return "\n".join(lines)


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
