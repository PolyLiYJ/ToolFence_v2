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


def reviewed_tools_for_context(
    benchmark: str,
    *,
    suite: str | None = None,
    server: str | None = None,
    agent: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_reviewed_manifest(benchmark)
    contexts = _contexts_for(benchmark, manifest)
    keys = [value for value in (suite, server, agent, "default") if value]
    context_key = next((key for key in keys if key in contexts), None)
    if context_key is None:
        context_key = "default" if "default" in contexts else None
    if context_key is None:
        raise ValueError(f"No reviewed manifest context for {benchmark}: {suite=} {server=} {agent=}")
    names = list(contexts[context_key]["tools"])
    catalog = []
    for name in names:
        spec = manifest["tools"][name]
        catalog.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", name),
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
        "denylist_count": len(manifest.get("denylist", [])),
    }
    return catalog, audit


def reviewed_semantics(benchmark: str, names: list[str]) -> dict[str, dict[str, Any]]:
    manifest = load_reviewed_manifest(benchmark)
    return {name: manifest["tools"][name] for name in names if name in manifest["tools"]}
