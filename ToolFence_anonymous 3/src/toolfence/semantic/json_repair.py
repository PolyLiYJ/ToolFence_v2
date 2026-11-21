"""Controller-side repair of malformed architect JSON.

Some hosted models (observed on qwen-max through both the official DashScope
compatible endpoint and a relay) emit the ``parameters`` field as a *string*
containing the object body:

    "parameters": "query: {\"semantic_role\": \"search_query\", ...}, sender: {...}"

where the contract requires a dictionary. The classification content is correct;
only the nesting is wrong. Deterministically repairing the shape keeps the
controller fail-closed: repair only re-wraps text that already parses as an
object, never invents parameter names, roles, or bindings, and never relaxes
authority (unparseable or unclassified parameters stay fail-closed).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


def coerce_object_text(text: str) -> Optional[Mapping[str, Any]]:
    """Re-wrap model text that is an object emitted without its braces.

    Returns None when the text cannot be read as an object. Only structural
    recovery is performed; no key or value is invented.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    while len(candidate) >= 2 and candidate[0] == '"' and candidate[-1] == '"':
        candidate = candidate[1:-1].strip()
    if not candidate:
        return None
    if candidate.startswith("{"):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed
    # Bare object body: "key: {...}, key2: {...}"
    if not candidate.startswith("{"):
        try:
            parsed = json.loads("{" + candidate + "}")
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed
    return _split_top_level(candidate)


def _split_top_level(text: str) -> Optional[Mapping[str, Any]]:
    """Split "k: v, k2: v2" on top-level commas while ignoring nested braces."""
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            buffer.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            buffer.append(char)
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)
    parts.append("".join(buffer))

    result: dict[str, Any] = {}
    for part in parts:
        key, separator, rest = part.partition(":")
        if not separator:
            continue
        name = key.strip().strip("\"'").strip()
        value_text = rest.strip()
        if not name:
            continue
        try:
            value = json.loads(value_text)
        except ValueError:
            value = value_text.strip("\"'")
        result[name] = value
    return result or None
