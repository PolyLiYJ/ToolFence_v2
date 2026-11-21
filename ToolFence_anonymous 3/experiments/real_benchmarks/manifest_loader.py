from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest_schema import manifest_hash, validate_manifest

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
_CACHE: dict[str, dict[str, Any]] = {}


def load_reviewed_manifest(benchmark: str) -> dict[str, Any]:
    if benchmark not in _CACHE:
        path = MANIFEST_DIR / f"{benchmark}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed manifest not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(data)
        data["_manifest_hash"] = manifest_hash(data)
        _CACHE[benchmark] = data
    return _CACHE[benchmark]


def _contexts_for(benchmark: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["contexts"]


def _context_key(
    contexts: dict[str, Any],
    *,
    suite: str | None = None,
    server: str | None = None,
    agent: str | None = None,
    allow_missing: bool = False,
) -> str | None:
    keys = [value for value in (suite, server, agent, "default") if value]
    context_key = next((key for key in keys if key in contexts), None)
    if context_key is None:
        context_key = "default" if "default" in contexts else None
    if context_key is None and not allow_missing:
        raise ValueError(f"No reviewed manifest context: {suite=} {server=} {agent=}")
    return context_key


def _bounded_description(value: Any, name: str, limit: int = 600) -> str:
    text = str(value or name).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _context_specs(
    manifest: dict[str, Any],
    context_key: str,
    names: list[str],
) -> dict[str, dict[str, Any]]:
    """Resolve tool names to context-specific specs.

    MCPTox has a few names shared by different servers with different schemas
    (for example Github and Commander both expose ``search_code``).  A context
    can carry ``tool_specs`` overrides for those collisions.
    """
    overrides = manifest["contexts"][context_key].get("tool_specs") or {}
    specs: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = overrides.get(name) or manifest["tools"].get(name)
        if spec is not None:
            specs[name] = spec
    return specs


def reviewed_tools_for_context(
    benchmark: str,
    *,
    suite: str | None = None,
    server: str | None = None,
    agent: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_reviewed_manifest(benchmark)
    contexts = _contexts_for(benchmark, manifest)
    context_key = _context_key(contexts, suite=suite, server=server, agent=agent)
    names = list(contexts[context_key]["tools"])
    specs = _context_specs(manifest, context_key, names)
    catalog = []
    for name in names:
        spec = specs.get(name)
        if spec is None:
            continue
        catalog.append({
            "type": "function",
            "function": {
                "name": spec.get("name", name),
                "description": _bounded_description(spec.get("description"), name),
                "parameters": spec.get("schema", {"type": "object", "properties": {}}),
            },
        })
    audit = {
        "benchmark": benchmark,
        "context": context_key,
        "manifest_hash": manifest["_manifest_hash"],
        "manifest_version": manifest["manifest_version"],
        "policy_revision": manifest["policy_revision"],
        "unknown_tool_policy": manifest["unknown_tool_policy"],
        "tool_count": len(catalog),
        "tool_names": names,
        "context_override_count": len(contexts[context_key].get("tool_specs") or {}),
        "denylist_count": len(manifest.get("denylist", [])),
    }
    return catalog, audit


def reviewed_semantics(
    benchmark: str,
    names: list[str],
    *,
    suite: str | None = None,
    server: str | None = None,
    agent: str | None = None,
) -> dict[str, dict[str, Any]]:
    manifest = load_reviewed_manifest(benchmark)
    contexts = _contexts_for(benchmark, manifest)
    context_key = _context_key(
        contexts, suite=suite, server=server, agent=agent, allow_missing=True
    )
    if context_key is None:
        return {
            name: manifest["tools"][name]
            for name in names
            if name in manifest["tools"]
        }
    return _context_specs(manifest, context_key, names)
