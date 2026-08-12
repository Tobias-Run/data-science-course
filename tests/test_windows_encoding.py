"""Guard against a real bug: text I/O without an explicit encoding.

Path.read_text()/write_text() default to locale.getpreferredencoding(), which
on Windows is a legacy codepage (commonly cp1252), not UTF-8. Every artefact
this project writes is JSON or JSON-backed pydantic output that can contain
arbitrary Unicode -- a non-ASCII prompt, a special character an LLM's plan text
happens to use -- and cp1252 cannot represent most of it. This surfaced for
real as a crash on the very first Windows run, on a single non-breaking hyphen
in a local model's response.

This is not a style nit tested for its own sake: every write_text/read_text
call in the package must pass encoding="utf-8" explicitly, and this test
enforces that mechanically so a future call site cannot reintroduce the bug
for whoever next runs on Windows.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _calls_without_utf8(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text")):
            continue
        has_encoding = any(kw.arg == "encoding" for kw in node.keywords) or (
            func.attr == "read_text" and len(node.args) >= 1
        ) or (func.attr == "write_text" and len(node.args) >= 2)
        if not has_encoding:
            bad.append(f"{path}:{node.lineno}: .{func.attr}() without encoding=")
    return bad


def test_all_text_io_specifies_utf8():
    root = Path(__file__).resolve().parents[1] / "worldclaw"
    violations = []
    for py_file in root.rglob("*.py"):
        violations += _calls_without_utf8(py_file)
    assert not violations, "missing explicit encoding (breaks on Windows):\n" + "\n".join(violations)
