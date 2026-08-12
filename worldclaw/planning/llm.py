"""Local LLM client for the planning stage.

Talks to any OpenAI-compatible endpoint -- LM Studio, Ollama, llama.cpp's
server, vLLM.  Only the standard library is used, so a local setup needs no
extra dependency and no vendor SDK.

The important feature is **schema-constrained decoding**: the endpoint is given
a JSON schema and the model physically cannot emit anything else.  For a
planning stage that has to produce a valid ``TerrainSpec`` this is the
difference between an agent that occasionally works and one that always does.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "http://localhost:1234/v1"


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``/``$defs`` so the schema is self-contained.

    Pydantic emits ``$ref`` into ``$defs`` for every nested model.  Server-side
    grammar compilers vary in how well they follow those references, and a
    half-followed reference degrades silently into a model that is free to emit
    anything.  Inlining costs nothing and removes the whole class of problem.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                if name in seen:  # recursive model: leave the ref, do not loop
                    return {"type": "object"}
                target = defs.get(name)
                if target is None:
                    return {"type": "object"}
                merged = walk(target, seen | {name})
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**merged, **extra} if extra else merged
            return {k: walk(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v, seen) for v in node]
        return node

    return walk({k: v for k, v in schema.items() if k != "$defs"}, frozenset())


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = "local-model"
    api_key: str = "lm-studio"  # LM Studio ignores it; other servers may not
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: float = 300.0


class LocalLLM:
    """Minimal OpenAI-compatible chat client with structured output."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def complete_json(
        self, system: str, user: str, model_cls: Type[T], schema_name: str | None = None
    ) -> T:
        """Ask for one object of ``model_cls`` and validate it before returning."""
        schema = inline_refs(model_cls.model_json_schema())
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or model_cls.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        text = self._post("/chat/completions", payload)
        try:
            content = text["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {text}") from exc
        try:
            return model_cls.model_validate_json(content)
        except Exception as exc:
            raise LLMError(
                f"{model_cls.__name__} validation failed. Model returned:\n{content[:2000]}"
            ) from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = self.config.base_url.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach {url}: {exc}\n"
                "Is LM Studio running with its local server started, and is the "
                "base URL right? (LM Studio default: http://localhost:1234/v1)"
            ) from exc

    def available_models(self) -> list[str]:
        """List what the endpoint serves -- the fastest connectivity check."""
        url = self.config.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.config.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return [m["id"] for m in json.loads(resp.read()).get("data", [])]
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach {url}: {exc}") from exc


@dataclass
class ScriptedLLM:
    """A stand-in that replays prepared objects instead of calling a model.

    Lets the whole planning stage be tested -- prompts, schema, expansion into a
    TerrainSpec, the layout gate -- without a GPU or a running server, which is
    the only way this stage is testable in CI at all.
    """

    responses: list[BaseModel] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete_json(
        self, system: str, user: str, model_cls: Type[T], schema_name: str | None = None
    ) -> T:
        self.calls.append((system, user))
        for i, r in enumerate(self.responses):
            if isinstance(r, model_cls):
                return self.responses.pop(i)  # type: ignore[return-value]
        raise LLMError(f"ScriptedLLM has no queued {model_cls.__name__}")
