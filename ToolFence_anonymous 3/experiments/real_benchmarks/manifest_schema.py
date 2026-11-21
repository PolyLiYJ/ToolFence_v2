from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

ALLOWED_EFFECTS = {
    "read", "local_compute", "communication", "reservation", "external_write",
    "financial", "delete", "credential_change", "unknown",
}
ALLOWED_OPERATIONS = {
    "observe", "compute", "communicate", "reserve", "create", "update",
    "delete", "financial", "credential_change", "unknown",
}
ALLOWED_GROUNDING_RULES = {
    "user_literal", "user_goal", "previous_read_output", "host_constant", "operational_default", "none",
}
REQUIRED_TOP_LEVEL = {
    "manifest_version", "benchmark", "policy_revision", "fail_closed",
    "unknown_tool_policy", "source_provenance", "contexts", "tools", "denylist",
}
REQUIRED_TOOL_FIELDS = {
    "name", "effect", "operation", "resource", "reversible", "reviewed_local_read",
    "external_disclosure", "mutates_state", "parameters", "host_adapter",
    "allowed_in_attack_context", "grounding_rule",
}


def manifest_hash(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_manifest(data: Any) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("manifest must be a JSON object")
    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if data["fail_closed"] is not True:
        raise ValueError("fail_closed must be true")
    if data["unknown_tool_policy"] != "deny":
        raise ValueError("unknown_tool_policy must be deny")
    if not isinstance(data.get("tools"), Mapping) or not data["tools"]:
        raise ValueError("tools must be a non-empty object")
    if not isinstance(data.get("contexts"), Mapping) or not data["contexts"]:
        raise ValueError("contexts must be a non-empty object")
    denylist = set(data.get("denylist") or [])
    for name, spec in data["tools"].items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"tool {name} must be an object")
        missing_fields = REQUIRED_TOOL_FIELDS - set(spec)
        if missing_fields:
            raise ValueError(f"tool {name} missing fields: {sorted(missing_fields)}")
        if spec["name"] != name:
            raise ValueError(f"tool key/name mismatch: {name} != {spec['name']}")
        if spec["effect"] not in ALLOWED_EFFECTS:
            raise ValueError(f"tool {name} invalid effect: {spec['effect']}")
        if spec["operation"] not in ALLOWED_OPERATIONS:
            raise ValueError(f"tool {name} invalid operation: {spec['operation']}")
        if spec["grounding_rule"] not in ALLOWED_GROUNDING_RULES:
            raise ValueError(f"tool {name} invalid grounding_rule: {spec['grounding_rule']}")
        params = spec.get("parameters") or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"tool {name} parameters must be an object")
        for param, rule in params.items():
            if not isinstance(rule, Mapping):
                raise ValueError(f"tool {name}.{param} must be an object")
            grounding = rule.get("grounding_rule", spec["grounding_rule"])
            if grounding not in ALLOWED_GROUNDING_RULES:
                raise ValueError(f"invalid grounding_rule for {name}.{param}: {grounding}")
    overlap = denylist & set(data["tools"])
    if overlap:
        raise ValueError(f"denylist overlaps reviewed tools: {sorted(overlap)}")
    for context, spec in data["contexts"].items():
        if not isinstance(spec, Mapping) or "tools" not in spec:
            raise ValueError(f"context {context} must contain tools")
        for name in spec["tools"]:
            if name not in data["tools"]:
                raise ValueError(f"context {context} references unknown tool {name}")
            if name in denylist:
                raise ValueError(f"context {context} references denied tool {name}")
