"""Guard against a real bug: text I/O without an explicit encoding.

Path.read_text()/write_text() and the builtin open() default to
locale.getpreferredencoding(), which on Windows is a legacy codepage (cp1252
here), not UTF-8. Every artefact this project writes is JSON, JSON-backed
pydantic output, or an OBJ header carrying a scene name -- all of it can
contain arbitrary Unicode, since it flows from prompts and local-model
responses. cp1252 cannot represent most of it.

This surfaced twice on the very first Windows run, on two different call
shapes: a Path.write_text() with no encoding argument (a non-breaking hyphen
in a model's plan text), and then a plain open(path, "w") with no encoding
argument (the same kind of hyphen, this time inside a model-chosen scene
name reaching the OBJ exporter). The first fix covered read_text/write_text;
this test now also covers plain open() calls in text mode, so the same class
of bug cannot resurface a third time through a call shape nobody thought to
check.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TEXT_MODE_CHARS = set("rwax")


def _open_is_binary(node: ast.Call) -> bool:
    """True if a builtin open() call's mode argument contains 'b'."""
    mode = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    if mode is None:
        return False  # default mode "r" is text, not binary
    return "b" in str(mode)


def _calls_without_utf8(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            has_encoding = any(kw.arg == "encoding" for kw in node.keywords) or (
                func.attr == "read_text" and len(node.args) >= 1
            ) or (func.attr == "write_text" and len(node.args) >= 2)
            if not has_encoding:
                bad.append(f"{path}:{node.lineno}: .{func.attr}() without encoding=")

        elif isinstance(func, ast.Name) and func.id == "open":
            if _open_is_binary(node):
                continue  # binary mode has no text encoding to get wrong
            has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
            if not has_encoding:
                bad.append(f"{path}:{node.lineno}: open() in text mode without encoding=")

    return bad


def test_all_text_io_specifies_utf8():
    root = Path(__file__).resolve().parents[1] / "worldclaw"
    violations = []
    for py_file in root.rglob("*.py"):
        violations += _calls_without_utf8(py_file)
    assert not violations, "missing explicit encoding (breaks on Windows):\n" + "\n".join(violations)
