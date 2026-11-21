"""LLM-backed semantic compiler with deterministic validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

from toolfence.foundation.blueprint import ParameterRule, SessionBlueprint, ToolBlueprint, ToolPermission

from .json_repair import coerce_object_text
from .prompts import (
    PLAN_COMPLETENESS_FEEDBACK,
    MULTI_STEP_COMPLETENESS_FEEDBACK,
    FINANCIAL_COMPLETENESS_FEEDBACK,
    PLAN_SYSTEM_PROMPT,
    SIDE_EFFECT_CUE_PATTERN,
    TOOL_SEMANTICS_SYSTEM_PROMPT,
    authorization_plan_prompt,
    tool_semantics_prompt,
)

# Binding kinds accepted by ParameterBinding. Used only to recognise a
# model-emitted shorthand such as "parameters": {"query": "literal"}.
BINDING_KINDS = frozenset({
    "literal", "normalized_literal", "template", "derived", "computed",
    "generated", "free",
})
from .schema import (
    AuthorizationPlan,
    Capability,
    ParameterBinding,
    ParameterSemantic,
    ToolManifest,
    ToolSemantic,
)


JSONBackend = Callable[[str, str], Mapping[str, Any] | str]
T = TypeVar("T")
READ_EFFECTS = {"read", "local_compute"}

# Controller-owned value shapes used only to recover bindings that a planner
# lost. Every recovered value must occur verbatim in the authenticated request.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://[^\s,;)\"']+|www\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,}[^\s,;)\"']*")
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(
    r"\b[\w\- ]+\.(?:txt|pdf|docx?|csv|md|json|ya?ml|png|jpe?g|xlsx?|pptx?)\b",
    re.IGNORECASE,
)

# Controller-owned vocabularies. A model value outside these sets is degraded
# to the fail-closed "unknown" classification instead of failing validation.
VALID_EFFECTS = frozenset({
    "read", "local_compute", "communication", "reservation", "financial",
    "credential_change", "external_write", "delete", "unknown",
})
VALID_OPERATIONS = frozenset({
    "observe", "compute", "communicate", "reserve", "create", "update",
    "delete", "financial", "credential_change", "unknown",
})
_QUOTED_RE = re.compile(r"'([^']{1,80})'|\"([^\"]{1,80})\"")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SIDE_EFFECTS = {
    "communication", "reservation", "financial", "credential_change",
    "external_write", "delete", "unknown",
}


@dataclass(frozen=True)
class SemanticCompileResult:
    blueprint: SessionBlueprint
    manifest: ToolManifest
    plan: AuthorizationPlan
    compiler_count: int = 2
    compiler_contractions: tuple[str, ...] = field(default_factory=tuple)


def _model_validate(model, payload):
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


def _model_dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _extract_json(value: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("Architect did not return a JSON object")
        # Some compatible endpoints append an explanation or a second JSON
        # value despite explicit JSON-only instructions. raw_decode returns the
        # first complete object without weakening its subsequent validation.
        payload, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, Mapping):
        raise ValueError("Architect JSON root must be an object")
    return payload


class OpenAICompatibleJSONBackend:
    """Small isolated client used by the policy architect."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_tokens: int = 2048,
    ):
        import openai

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.json_mode_fallbacks = 0
        self.observed_model_ids: set[str] = set()
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        request = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        if self.model.lower().startswith("gpt-5"):
            request["max_completion_tokens"] = self.max_tokens
        else:
            request["temperature"] = 0.0
            request["max_tokens"] = self.max_tokens
            # Local reasoning endpoints such as Qwen accept this extension;
            # OpenAI's public API should receive only documented fields.
            if "siliconflow.cn" in self.base_url:
                if self.model in {"Qwen/Qwen3-8B", "Qwen/Qwen3.5-35B-A3B"}:
                    request["extra_body"] = {"enable_thinking": False}
            elif "api.openai.com" not in self.base_url and "generativelanguage.googleapis.com" not in self.base_url:
                request["extra_body"] = {"think": False}
        self.request_count += 1
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.failed_count += 1
            if (getattr(exc, "status_code", None) != 400
                    or "Failed to initialize samplers" not in str(exc)):
                raise
            # Some llama.cpp builds reject json_object before generation. The
            # compiler still parses and validates the returned JSON schema;
            # only the transport-level formatting hint is removed on retry.
            request.pop("response_format")
            self.json_mode_fallbacks += 1
            self.request_count += 1
            try:
                response = self.client.chat.completions.create(**request)
            except Exception:
                self.failed_count += 1
                raise
        self.success_count += 1
        if getattr(response, "model", None):
            self.observed_model_ids.add(str(response.model))
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Architect returned empty content")
        return content

    def audit_snapshot(self) -> dict[str, object]:
        return {
            "requested_model_id": self.model,
            "observed_model_ids": sorted(self.observed_model_ids),
            "api_call_count": self.request_count,
            "successful_api_calls": self.success_count,
            "failed_api_calls": self.failed_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "json_mode_fallbacks": self.json_mode_fallbacks,
            "temperature": 0.0 if not self.model.lower().startswith("gpt-5") else None,
            "max_output_tokens": self.max_tokens,
            "endpoint": self.base_url,
        }


class SemanticPolicyCompiler:
    """Compile model-proposed JSON into a fail-closed immutable blueprint."""

    def __init__(
        self,
        backend: JSONBackend,
        *,
        max_compile_attempts: int = 3,
        read_tool_budget: int = 8,
        session_read_budget: int = 32,
        plural_side_effect_budget: int = 8,
        read_tools_open: bool = False,
        completeness_second_pass: bool = False,
        pinned_manifest: ToolManifest | None = None,
    ):
        self.backend = backend
        if max_compile_attempts < 1:
            raise ValueError("max_compile_attempts must be positive")
        if read_tool_budget < 1 or session_read_budget < 1 or plural_side_effect_budget < 1:
            raise ValueError("read budgets must be positive")
        self.max_compile_attempts = max_compile_attempts
        self.read_tool_budget = int(read_tool_budget)
        self.session_read_budget = int(session_read_budget)
        self.plural_side_effect_budget = int(plural_side_effect_budget)
        # v3 options: controller-issued read defaults and a bounded planner
        # completeness re-check. Both keep side-effect authorization fail-closed.
        self.read_tools_open = bool(read_tools_open)
        self.completeness_second_pass = bool(completeness_second_pass)
        # The ToolFence profile supplies reviewed effect/parameter metadata.
        # Legacy benchmark profiles may still compile this manifest with a model.
        self.pinned_manifest = (
            pinned_manifest.model_copy(deep=True) if pinned_manifest is not None else None
        )
        self._manifest_cache: Dict[str, ToolManifest] = {}

    def _generate_validated(
        self,
        *,
        label: str,
        system_prompt: str,
        user_prompt: str,
        validator: Callable[[Mapping[str, Any]], T],
    ) -> T:
        """Generate, validate, and ask the same isolated model to repair JSON.

        The repair loop is schema-driven: it receives only the original trusted
        compiler input, its own invalid output, and the validation error. It does
        not inspect tool observations and contains no task-, suite-, or tool-name
        special cases.
        """
        prompt = user_prompt
        errors: List[str] = []
        for attempt in range(1, self.max_compile_attempts + 1):
            raw = self.backend(system_prompt, prompt)
            try:
                return validator(_extract_json(raw))
            except (TypeError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {exc}")
                if attempt == self.max_compile_attempts:
                    break
                if isinstance(raw, str):
                    invalid_output = raw
                else:
                    invalid_output = json.dumps(raw, ensure_ascii=False, default=str)
                prompt = json.dumps({
                    "task": f"repair_invalid_{label}",
                    "original_request": user_prompt,
                    "invalid_output": invalid_output,
                    "validation_error": str(exc),
                    "instruction": (
                        "Return one complete replacement JSON object satisfying "
                        "the original output_contract. Do not return a fragment, "
                        "an explanation, Markdown, or multiple JSON objects."
                    ),
                }, ensure_ascii=False, sort_keys=True)
        raise ValueError(
            f"{label} compilation failed after {self.max_compile_attempts} attempts: "
            + " | ".join(errors)
        )

    @staticmethod
    def _compact_parameter_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
        """Retain authorization-relevant schema facts without verbose Pydantic defs."""
        kept = {}
        for key in (
            "type", "description", "enum", "const", "format", "pattern",
            "minimum", "maximum", "minLength", "maxLength",
        ):
            if key in schema:
                kept[key] = schema[key]
        variants = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(variants, list):
            kept["variants"] = [
                SemanticPolicyCompiler._compact_parameter_schema(item)
                for item in variants if isinstance(item, Mapping)
            ]
            if "type" not in kept:
                concrete_types = [
                    item.get("type") for item in variants
                    if isinstance(item, Mapping) and item.get("type") not in {None, "null"}
                ]
                if concrete_types:
                    kept["type"] = concrete_types[0]
        if schema.get("items") and isinstance(schema["items"], Mapping):
            kept["items"] = SemanticPolicyCompiler._compact_parameter_schema(schema["items"])
        return kept

    @staticmethod
    def _compact_object_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
        properties = schema.get("properties", {})
        def resolve(raw: Mapping[str, Any]) -> Mapping[str, Any]:
            ref = raw.get("$ref")
            if not isinstance(ref, str) or not ref.startswith("#/"):
                return raw
            parts = ref[2:].split("/")
            target: Any = schema
            for part in parts:
                if not isinstance(target, Mapping) or part not in target:
                    return raw
                target = target[part]
            if not isinstance(target, Mapping):
                return raw
            return {**target, **{key: value for key, value in raw.items() if key != "$ref"}}

        return {
            "type": "object",
            "properties": {
                str(name): SemanticPolicyCompiler._compact_parameter_schema(resolve(raw))
                for name, raw in properties.items() if isinstance(raw, Mapping)
            },
            "required": [
                str(name) for name in schema.get("required", []) if name in properties
            ],
        }

    @staticmethod
    def _normalise_tool(raw: Mapping[str, Any]) -> Dict[str, Any]:
        tool = dict(raw.get("function", raw))
        name = tool.get("name")
        if not name:
            raise ValueError("Every tool must have a name")
        parameters = tool.get("parameters", tool.get("params", {})) or {}
        return {
            "name": str(name),
            "description": str(tool.get("description", "")),
            "parameters": SemanticPolicyCompiler._compact_object_schema(parameters),
        }

    @staticmethod
    def _catalog_key(tools: Sequence[Mapping[str, Any]]) -> str:
        raw = json.dumps(list(tools), sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _default_operation(effect: str) -> str:
        return {
            "read": "observe",
            "local_compute": "compute",
            "communication": "communicate",
            "reservation": "reserve",
            "financial": "financial",
            "credential_change": "credential_change",
            "external_write": "update",
            "delete": "delete",
        }.get(effect, "unknown")

    @staticmethod
    def _schema_property_names(tool: Optional[Mapping[str, Any]]) -> Optional[set]:
        if not isinstance(tool, Mapping):
            return None
        properties = tool.get("parameters", {}).get("properties", {})
        return set(properties) if isinstance(properties, Mapping) else None

    @staticmethod
    def _repair_manifest_payload(
        payload: Mapping[str, Any], log: Optional[list],
        catalog: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Repair object-shaped fields emitted as strings (qwen-max quirk)."""
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return payload
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name", "?")
            # Drop keys that are not real parameter names of this tool. Small
            # planners sometimes emit the schema template itself (for example
            # {"authority_sensitive": {}}) instead of parameter entries.
            allowed = (
                SemanticPolicyCompiler._schema_property_names((catalog or {}).get(name))
                if catalog else None
            )
            if allowed is not None and isinstance(tool.get("parameters"), dict):
                dropped = [
                    key for key in tool["parameters"] if key not in allowed
                ]
                for key in dropped:
                    tool["parameters"].pop(key, None)
                if dropped and log is not None:
                    log.append(f"manifest_parameters_filtered:{name}")
            parameters = tool.get("parameters")
            if isinstance(parameters, str):
                repaired = coerce_object_text(parameters)
                tool["parameters"] = repaired if repaired is not None else {}
                if log is not None:
                    log.append(
                        f"manifest_parameters_repaired:{name}"
                        if repaired is not None else f"manifest_parameters_emptied:{name}"
                    )
            parameters = tool.get("parameters")
            if not isinstance(parameters, dict):
                continue
            effect = tool.get("effect")
            for param_name, value in list(parameters.items()):
                if isinstance(value, str):
                    # A bare string carries a role only; default sensitivity is
                    # fail-closed for side-effect tools.
                    parameters[param_name] = {
                        "semantic_role": value,
                        "authority_sensitive": effect not in READ_EFFECTS,
                    }
                    if log is not None:
                        log.append(f"manifest_parameter_role_wrapped:{name}.{param_name}")
                    continue
                if isinstance(value, dict):
                    role = value.get("semantic_role")
                    sensitive = value.get("authority_sensitive")
                    repaired = {
                        "semantic_role": role if isinstance(role, str) else "unclassified",
                        # Non-boolean or missing sensitivity fails closed.
                        "authority_sensitive": (
                            sensitive if isinstance(sensitive, bool)
                            else effect not in READ_EFFECTS
                        ),
                    }
                    if repaired != value:
                        parameters[param_name] = repaired
                        if log is not None:
                            log.append(f"manifest_parameter_normalized:{name}.{param_name}")
                    continue
                parameters[param_name] = {
                    "semantic_role": "unclassified",
                    "authority_sensitive": True,
                }
                if log is not None:
                    log.append(f"manifest_parameter_defaulted:{name}.{param_name}")
            # Enumerated fields: an unusable value becomes the fail-closed
            # "unknown" classification rather than a validation error.
            if tool.get("effect") not in VALID_EFFECTS:
                tool["effect"] = "unknown"
            if tool.get("operation") not in VALID_OPERATIONS:
                tool["operation"] = "unknown"
            if not isinstance(tool.get("resource"), str):
                tool["resource"] = "unknown"
            if not isinstance(tool.get("reversible"), bool):
                tool["reversible"] = False
        return payload

    @staticmethod
    def _repair_plan_payload(
        payload: Mapping[str, Any], log: Optional[list],
        catalog: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Repair capability parameter blocks emitted as strings."""
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, dict):
            capabilities = [capabilities]
            payload["capabilities"] = capabilities
        if isinstance(capabilities, str):
            repaired_all = coerce_object_text(capabilities)
            capabilities = [repaired_all] if repaired_all is not None else []
            payload["capabilities"] = capabilities
        if not isinstance(capabilities, list):
            return payload
        for index, capability in enumerate(capabilities):
            if isinstance(capability, str):
                repaired = coerce_object_text(capability)
                capabilities[index] = repaired if repaired is not None else {}
                if log is not None:
                    log.append("plan_capability_repaired")
                capability = capabilities[index]
            if not isinstance(capability, dict):
                continue
            tool_name = capability.get("tool", "?")
            if not tool_name:
                # An unrecoverable capability fragment carries no tool identity;
                # drop it instead of validating an empty object.
                capabilities.remove(capability)
                if log is not None:
                    log.append("plan_capability_fragment_dropped")
                continue
            parameters = capability.get("parameters")
            if isinstance(parameters, str):
                repaired = coerce_object_text(parameters)
                capability["parameters"] = repaired if repaired is not None else {}
                if log is not None:
                    log.append(
                        f"plan_parameters_repaired:{tool_name}"
                        if repaired is not None else f"plan_parameters_emptied:{tool_name}"
                    )
            parameters = capability.get("parameters")
            if not isinstance(parameters, dict):
                continue
            allowed = (
                SemanticPolicyCompiler._schema_property_names((catalog or {}).get(tool_name))
                if catalog else None
            )
            if allowed is not None:
                dropped = [key for key in parameters if key not in allowed]
                for key in dropped:
                    parameters.pop(key, None)
                if dropped and log is not None:
                    log.append(f"plan_parameters_filtered:{tool_name}")
            for param_name, binding in list(parameters.items()):
                if isinstance(binding, str):
                    text = binding.strip()
                    if text in BINDING_KINDS:
                        parameters[param_name] = {"binding": text}
                    else:
                        repaired = coerce_object_text(text)
                        parameters[param_name] = repaired or {}
                    if log is not None:
                        log.append(f"plan_binding_wrapped:{tool_name}.{param_name}")
                    continue
                if isinstance(binding, dict):
                    kind = binding.get("binding")
                    if kind not in BINDING_KINDS:
                        # An unusable binding never invents authority: the
                        # parameter is dropped and stays undeclared (denied).
                        parameters.pop(param_name, None)
                        if log is not None:
                            log.append(f"plan_binding_dropped:{tool_name}.{param_name}")
                    continue
                parameters.pop(param_name, None)
                if log is not None:
                    log.append(f"plan_binding_dropped:{tool_name}.{param_name}")
        return payload

    def compile_manifest(self, tools: Sequence[Mapping[str, Any]]) -> ToolManifest:
        normalized = [self._normalise_tool(tool) for tool in tools]
        if self.pinned_manifest is not None:
            catalog = {tool["name"]: tool for tool in normalized}
            pinned = self.pinned_manifest.by_name()
            if len(catalog) != len(normalized) or len(pinned) != len(self.pinned_manifest.tools):
                raise ValueError("Duplicate tool in trusted catalog or pinned manifest")
            if set(catalog) != set(pinned):
                raise ValueError("Pinned manifest must cover exactly the trusted catalog")
            for name, semantic in pinned.items():
                fields = set(catalog[name]["parameters"].get("properties", {}))
                if set(semantic.parameters) != fields:
                    raise ValueError("Pinned parameter classifications must match tool schema")
                if semantic.effect == "unknown" or semantic.operation == "unknown":
                    raise ValueError("Pinned operation and effect must be explicit")
            return self.pinned_manifest.model_copy(deep=True)
        cache_key = self._catalog_key(normalized)
        if cache_key in self._manifest_cache:
            return self._manifest_cache[cache_key]
        catalog = {tool["name"]: tool for tool in normalized}
        if len(catalog) != len(normalized):
            raise ValueError("Controller catalog contains duplicate tool names")

        def validate(payload: Mapping[str, Any]) -> ToolManifest:
            repair_log: list = []
            payload = self._repair_manifest_payload(payload, repair_log, catalog)
            proposed = _model_validate(ToolManifest, payload)
            seen = set()
            normalized_semantics = []
            for semantic in proposed.tools:
                if semantic.name not in catalog or semantic.name in seen:
                    raise ValueError(f"Invalid or duplicate manifest tool: {semantic.name}")
                seen.add(semantic.name)
                properties = catalog[semantic.name]["parameters"].get("properties", {})
                # Unknown model-produced fields are a safe contraction: they can
                # never occur in a controller-owned tool schema and therefore
                # cannot grant an executable capability.
                parameters = {
                    name: value for name, value in semantic.parameters.items()
                    if name in properties
                }
                missing = set(properties) - set(semantic.parameters)
                # Missing classifications are filled fail-closed. They may be
                # used only with an explicit literal/derived binding; free or
                # generated authority remains forbidden.
                for name in missing:
                    parameters[name] = ParameterSemantic(
                        semantic_role="unclassified",
                        authority_sensitive=True,
                    )
                operation = semantic.operation
                if operation == "unknown" and semantic.effect in SIDE_EFFECTS and semantic.effect != "unknown":
                    raise ValueError(
                        f"Side-effect tool {semantic.name} requires an explicit primary operation"
                    )
                if operation == "unknown":
                    operation = self._default_operation(semantic.effect)
                normalized_semantics.append(ToolSemantic(
                    name=semantic.name,
                    effect=semantic.effect,
                    operation=operation,
                    resource=semantic.resource,
                    reversible=semantic.reversible,
                    parameters=parameters,
                ))
            missing_tools = set(catalog) - seen
            if missing_tools:
                raise ValueError(
                    "Tool manifest must classify every available tool exactly once; "
                    f"missing={sorted(missing_tools)}"
                )
            return ToolManifest(tools=normalized_semantics)

        proposed = self._generate_validated(
            label="tool_manifest",
            system_prompt=TOOL_SEMANTICS_SYSTEM_PROMPT,
            user_prompt=tool_semantics_prompt(normalized),
            validator=validate,
        )
        self._manifest_cache[cache_key] = proposed
        return proposed

    @staticmethod
    def _literal_supported(
        query: str,
        value: Any,
        *,
        parameter_name: str = "",
        semantic_role: str = "",
    ) -> bool:
        normalized_query = " ".join(query.casefold().split())
        if isinstance(value, bool):
            semantic_terms = re.findall(
                r"[a-z0-9]+", f"{parameter_name} {semantic_role}".casefold()
            )
            return any(len(term) >= 4 and term in normalized_query for term in semantic_terms)
        scalars: List[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, Mapping):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (list, tuple, set)):
                for child in item:
                    collect(child)
            elif item is not None:
                scalars.append(" ".join(str(item).casefold().split()))

        collect(value)
        return bool(scalars) and all(item and item in normalized_query for item in scalars)


    @staticmethod
    def _normalized_literal_grounding(
        query: str, value: Any,
    ) -> Optional[tuple[int, int, str, List[str]]]:
        """Ground a tightly bounded normalized literal in authenticated text.

        Month/day (and an explicitly requested time) must be authenticated. If
        the user omits a calendar year, exactly one year chosen by the pre-
        execution planner may be fixed into the capability and is recorded as
        ``inferred_year``. This is not a wildcard: runtime still accepts only
        that exact compiled value. A missing time may normalize only to midnight
        and is recorded as ``default_midnight``.
        """
        if not isinstance(value, str) or not query:
            return None
        target = value.strip()
        if not target:
            return None

        idx = query.casefold().find(target.casefold())
        if idx >= 0:
            return idx, idx + len(target), query[idx:idx + len(target)], []

        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
            "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12,
        }

        date_match = re.search(
            r"(?<!\d)(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})(?!\d)",
            target,
        )
        time_match = re.search(
            r"(?<!\d)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?(?!\d)",
            target,
        )

        def month_pattern(month: int) -> str:
            names = [name for name, number in months.items() if number == month]
            return "(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + ")"

        if date_match:
            year = int(date_match.group("year"))
            month = int(date_match.group("month"))
            day = int(date_match.group("day"))
            mp = month_pattern(month)
            ordinal = rf"{day}(?:st|nd|rd|th)?"
            explicit_year = bool(re.search(rf"(?<!\d){year}(?!\d)", query))
            year_part = rf"(?:,)?\s+{year}" if explicit_year else ""
            patterns = [
                rf"\b{mp}\s+{ordinal}{year_part}\b",
                rf"\b{ordinal}(?:\s+of)?\s+{mp}{year_part}\b",
            ]
            if explicit_year:
                patterns.extend([
                    rf"(?<!\d){year}[-/]0?{month}[-/]0?{day}(?!\d)",
                    rf"(?<!\d)0?{month}[-/]0?{day}[-/]{year}(?!\d)",
                ])
            for pattern in patterns:
                m = re.search(pattern, query, flags=re.IGNORECASE)
                if not m:
                    continue
                inferred: List[str] = []
                if not explicit_year:
                    inferred.append("inferred_year")
                if time_match:
                    hour = int(time_match.group("hour"))
                    minute = int(time_match.group("minute"))
                    second = int(time_match.group("second") or 0)
                    # If time is absent from the user's request, only the neutral
                    # midnight default may be introduced.
                    fragment = query[m.start():m.end()]
                    if not SemanticPolicyCompiler._time_components_supported(query, hour, minute):
                        if (hour, minute, second) != (0, 0, 0):
                            return None
                        inferred.append("default_midnight")
                return m.start(), m.end(), query[m.start():m.end()], inferred

        if time_match:
            hour24 = int(time_match.group("hour"))
            minute = int(time_match.group("minute"))
            for hour, suffix in (
                (hour24, ""),
                (12 if hour24 == 0 else hour24 if hour24 <= 12 else hour24 - 12,
                 "am" if hour24 < 12 else "pm"),
            ):
                pattern = (
                    rf"(?<!\d){hour}(?::0?{minute})?\s*{suffix}\b"
                    if suffix else rf"(?<!\d){hour}:0?{minute}(?!\d)"
                )
                m = re.search(pattern, query, flags=re.IGNORECASE)
                if m:
                    return m.start(), m.end(), query[m.start():m.end()], []

        return None

    @staticmethod
    def _time_components_supported(query: str, hour24: int, minute: int) -> bool:
        candidates = [rf"(?<!\d){hour24}:0?{minute}(?!\d)"]
        if hour24 == 0:
            hour12, suffix = 12, "am"
        elif hour24 < 12:
            hour12, suffix = hour24, "am"
        elif hour24 == 12:
            hour12, suffix = 12, "pm"
        else:
            hour12, suffix = hour24 - 12, "pm"
        candidates.append(rf"(?<!\d){hour12}(?::0?{minute})?\s*{suffix}\b")
        return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in candidates)

    @staticmethod
    def _side_effect_action_cues(operation: str, effect: str) -> tuple[str, ...]:
        by_operation = {
            # Keep these operation-level rather than tool-name-specific. The
            # controller still requires an authenticated user span and rejects
            # explicit negation; these extra verbs only avoid lexical false
            # negatives such as "post", "reply", "invite", and "add".
            "communicate": (
                "send", "email", "message", "share", "notify", "forward",
                "deliver", "post", "reply", "write", "tell",
            ),
            "reserve": ("book", "reserve", "reservation", "schedule"),
            "create": (
                "create", "add", "schedule", "write", "save", "post",
                "invite", "register",
            ),
            "update": (
                "update", "change", "set", "edit", "write", "save", "add",
                "join", "assign", "move", "share", "grant",
            ),
            "delete": ("delete", "remove", "cancel"),
            "financial": ("pay", "payment", "refund", "transfer", "purchase", "buy"),
            # Some access-control tools are conservatively classified as
            # credential_change by small manifest models. Explicit share/grant
            # requests still ground their primary access update operation.
            "credential_change": (
                "reset", "change", "update", "set", "share", "grant",
            ),
        }
        if operation in by_operation:
            return by_operation[operation]
        return {
            "communication": by_operation["communicate"],
            "reservation": by_operation["reserve"],
            "financial": by_operation["financial"],
            "credential_change": by_operation["credential_change"],
            "external_write": tuple(dict.fromkeys(by_operation["create"] + by_operation["update"])),
            "delete": by_operation["delete"],
        }.get(effect, ())

    @staticmethod
    def _resource_terms(resource: str) -> set[str]:
        terms = set(re.findall(r"[a-z0-9]+", resource.casefold()))
        return {
            term for term in terms
            if len(term) >= 3 and term not in {"unknown", "resource", "record", "object"}
        }

    @staticmethod
    def _explicitly_negated(text: str, cue: str) -> bool:
        tokens = re.findall(r"[a-z0-9']+", text.casefold())
        cue_tokens = re.findall(r"[a-z0-9']+", cue.casefold())
        if not cue_tokens:
            return False
        first = cue_tokens[0]
        for index, token in enumerate(tokens):
            if token != first:
                continue
            window = tokens[max(0, index - 4):index]
            if any(item in {"not", "never", "without", "dont", "don't", "no"} for item in window):
                return True
            if len(window) >= 2 and window[-2:] == ["do", "not"]:
                return True
        return False

    @staticmethod
    def _authenticated_value_for(
        query: str, name: str, parameter: Any, schema: Mapping[str, Any],
    ) -> Optional[Any]:
        """Return the single unambiguous authenticated value for a parameter.

        Only values that literally occur in the authenticated request qualify.
        Ambiguity (zero or several distinct candidates) fails closed and leaves
        the parameter undeclared, so no authority is invented.
        """
        label = f"{name} {getattr(parameter, 'semantic_role', '') or ''}".casefold()
        schema_type = (schema or {}).get("type", "string")
        candidates: list = []
        if re.search(r"recipient|\bto\b|email|cc|bcc|sender|from|user|member|invite|contact", label):
            candidates = _EMAIL_RE.findall(query)
        elif re.search(r"url|link|webpage|website|host|domain|href", label):
            candidates = _URL_RE.findall(query)
        elif re.search(r"date|time|day|deadline|start|end|schedule", label):
            candidates = _DATE_RE.findall(query)
        elif re.search(r"file|document|filename|path|attachment|folder", label):
            candidates = _FILENAME_RE.findall(query) + [
                group for pair in _QUOTED_RE.findall(query) for group in pair if group
            ]
        elif re.search(r"amount|price|sum|total|count|number|balance|quantity", label):
            candidates = _NUMBER_RE.findall(query)
        else:
            candidates = [
                group for pair in _QUOTED_RE.findall(query) for group in pair if group
            ]
        unique = list(dict.fromkeys(item for item in candidates if item))
        if len(unique) != 1:
            return None
        value = unique[0]
        return [value] if schema_type == "array" else value

    @staticmethod
    def _recover_bindings(
        query: str,
        tool: str,
        semantic: ToolSemantic,
        properties: Mapping[str, Any],
        contractions: Optional[List[str]],
    ) -> Dict[str, ParameterBinding]:
        """Deterministically rebuild lost parameter bindings (fail-closed)."""
        bindings: Dict[str, ParameterBinding] = {}
        for name, schema in properties.items():
            parameter = semantic.parameters.get(name)
            sensitive = parameter.authority_sensitive if parameter is not None else True
            if not sensitive:
                bindings[name] = ParameterBinding(binding="free")
                continue
            value = SemanticPolicyCompiler._authenticated_value_for(
                query, name, parameter, schema or {},
            )
            if value is None:
                continue
            bindings[name] = ParameterBinding(
                binding="literal", value=value, evidence=str(value),
            )
        if bindings and contractions is not None:
            contractions.append(
                f"controller_binding_recovered:{tool}:"
                + ",".join(sorted(bindings))
            )
        return bindings

    @staticmethod
    def _controller_ground_side_effect(
        query: str, capability: Capability, semantic: ToolSemantic,
    ) -> Optional[tuple[int, int, str]]:
        """Return a controller-owned authenticated span authorizing a side effect.

        The planner may propose evidence text or offsets, but neither is trusted.
        We accept only a slice that exists in the authenticated query and contains
        an effect-level action cue plus, when available, a semantic resource term.
        This avoids tool-name/task-specific allowlists while preventing mere object
        mentions from authorizing side effects.
        """
        if not query:
            return None

        action_cues = SemanticPolicyCompiler._side_effect_action_cues(
            semantic.operation, semantic.effect,
        )
        resource_terms = SemanticPolicyCompiler._resource_terms(semantic.resource)

        def supported(fragment: str) -> bool:
            normalized = " ".join(fragment.casefold().split())
            cue = next((item for item in action_cues if re.search(rf"\b{re.escape(item)}\w*\b", normalized)), None)
            if cue is None or SemanticPolicyCompiler._explicitly_negated(normalized, cue):
                return False
            # Resource overlap remains a useful conservative signal, but some
            # verbs already encode the side-effect semantics themselves (e.g.
            # "invite", "reply", "post", "refund").  Requiring the manifest's
            # synthetic resource noun as well caused false negatives such as
            # "post there" versus resource="message".  Accept either explicit
            # resource overlap or a strong operation-specific action verb.
            words = set(re.findall(r"[a-z0-9]+", normalized))
            if not resource_terms or words & resource_terms:
                return True
            strong_cues = {
                "invite", "register", "post", "reply", "email", "message",
                "notify", "forward", "deliver", "book", "reserve",
                "reservation", "create", "add", "join", "assign", "move",
                "update", "edit", "share", "grant", "delete", "remove", "cancel", "refund",
                "transfer", "purchase", "buy", "pay",
            }
            return cue in strong_cues

        # 0) v4: when the planner omitted evidence entirely (observed on qwen-max,
        # which degrades nested plan objects), ground the side effect on the
        # smallest sentence of the authenticated query that carries the action
        # cue. The trust source is unchanged: only authenticated user text.
        if not capability.evidence.strip() and not (
            isinstance(capability.evidence_start, int)
            and isinstance(capability.evidence_end, int)
        ):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", query):
                fragment = sentence.strip()
                if fragment and supported(fragment):
                    start = query.find(fragment)
                    return start, start + len(fragment), fragment

        # 1) Validate planner-proposed offsets only by slicing controller-owned text.
        start, end = capability.evidence_start, capability.evidence_end
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(query):
            fragment = query[start:end]
            if supported(fragment):
                return start, end, fragment

        # 2) Backwards-compatible evidence string: locate it in the authenticated query.
        evidence = capability.evidence.strip()
        if evidence:
            lower_query = query.casefold()
            idx = lower_query.find(evidence.casefold())
            if idx >= 0:
                fragment = query[idx:idx + len(evidence)]
                if supported(fragment):
                    return idx, idx + len(evidence), fragment

        # 3) Deterministic fallback over authenticated clauses. This is controller-
        # grounded and effect/resource based; tool outputs never participate.
        for match in re.finditer(r"[^.!?;\n]+[.!?;]?", query):
            fragment = match.group(0).strip()
            if fragment and supported(fragment):
                raw = match.group(0)
                leading = len(raw) - len(raw.lstrip())
                trailing = len(raw.rstrip())
                frag_start = match.start() + leading
                frag_end = match.start() + trailing
                return frag_start, frag_end, query[frag_start:frag_end]
        return None

    @staticmethod
    def _query_requests_semantic_transform(query: str) -> bool:
        """Return True for authenticated requests that explicitly ask to transform data.

        This is consulted only for non-authority-sensitive sink parameters. It
        never relaxes recipients, targets, resources, accounts, amounts, dates,
        or other authority-sensitive bindings.
        """
        text = " ".join(query.casefold().split())
        cues = (
            "summarize", "summary", "feedback", "based on", "what kind",
            "explain", "describe", "calculate", "compute", "total",
            "recommend", "recommendation", "translate", "rewrite",
        )
        return any(cue in text for cue in cues)

    @staticmethod
    def _is_temporal_parameter(
        name: str, parameter: Optional[ParameterSemantic],
    ) -> bool:
        """Identify temporal slots from controller-owned names and semantics."""
        role = parameter.semantic_role if parameter is not None else ""
        tokens = set(re.findall(r"[a-z0-9]+", f"{name} {role}".casefold()))
        return bool(tokens & {
            "date", "day", "time", "datetime", "timestamp", "temporal",
            "deadline", "startdate", "enddate", "starttime", "endtime",
        })

    @staticmethod
    def _permission_enum_alias(
        query: str,
        value: Any,
        enum: Sequence[Any],
        name: str,
        parameter: Optional[ParameterSemantic],
    ) -> Optional[tuple[Any, int, int, str]]:
        """Normalize common access-level words to compact schema enum values."""
        if not isinstance(value, str):
            return None
        role = parameter.semantic_role if parameter is not None else ""
        tokens = set(re.findall(r"[a-z0-9]+", f"{name} {role}".casefold()))
        if not tokens & {"permission", "permissions", "access", "level"}:
            return None
        aliases = {
            "r": ("read-only", "read only", "read"),
            "rw": ("read-write", "read write", "write", "edit"),
        }
        normalized_value = value.casefold().strip()
        for enum_value in enum:
            enum_text = str(enum_value).casefold()
            phrases = aliases.get(enum_text, (enum_text,))
            if normalized_value != enum_text and normalized_value not in phrases:
                continue
            for phrase in phrases:
                match = re.search(rf"\b{re.escape(phrase)}\b", query, re.I)
                if match:
                    return enum_value, match.start(), match.end(), query[match.start():match.end()]
        return None

    @staticmethod
    def _template_grounding(query: str, value: Any) -> Optional[tuple[int, int, str]]:
        """Ground an authenticated presentation template such as ``Hotel: {name}``."""
        if not isinstance(value, str) or not value or "{" not in value or "}" not in value:
            return None
        idx = query.casefold().find(value.casefold())
        if idx < 0:
            return None
        return idx, idx + len(value), query[idx:idx + len(value)]

    @staticmethod
    def _synthesize_read_dependency(
        query: str,
        tool_name: str,
        semantic: ToolSemantic,
        catalog: Mapping[str, Mapping[str, Any]],
    ) -> Optional[Capability]:
        """Complete a missing read dependency without inventing sensitive authority.

        A planner may correctly state that a value must come from a read tool but
        omit the source tool's own capability.  The controller may synthesize the
        missing source only when every required parameter is either grounded by an
        unambiguous enum literal in the authenticated request or is explicitly
        classified non-authority-sensitive.  Sensitive required parameters are
        never guessed.
        """
        if semantic.effect not in READ_EFFECTS or tool_name not in catalog:
            return None
        schema = catalog[tool_name]["parameters"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        bindings: Dict[str, ParameterBinding] = {}
        for name in required:
            parameter_semantic = semantic.parameters.get(name)
            authority_sensitive = (
                True if parameter_semantic is None else parameter_semantic.authority_sensitive
            )
            raw = properties.get(name, {})
            enum = raw.get("enum") or []
            grounded_values = [
                value for value in enum
                if SemanticPolicyCompiler._literal_supported(
                    query,
                    value,
                    parameter_name=name,
                    semantic_role=(
                        parameter_semantic.semantic_role if parameter_semantic else ""
                    ),
                )
            ]
            if len(grounded_values) == 1:
                bindings[name] = ParameterBinding(
                    binding="literal", value=grounded_values[0],
                )
                continue
            if authority_sensitive:
                return None
            bindings[name] = ParameterBinding(binding="free")
        return Capability(
            tool=tool_name,
            effect=semantic.effect,
            operation=semantic.operation,
            reusable=True,
            origin="controller_dependency",
            max_calls=1,
            parameters=bindings,
        )

    @staticmethod
    def _validate_plan(
        query: str,
        plan: AuthorizationPlan,
        manifest: ToolManifest,
        tools: Sequence[Mapping[str, Any]],
        contractions: Optional[List[str]] = None,
        plural_side_effect_budget: int = 8,
        strict_fields: bool = False,
    ) -> AuthorizationPlan:
        catalog = {tool["name"]: tool for tool in tools}
        semantics = manifest.by_name()
        accepted: List[Capability] = []
        for capability in plan.capabilities:
            if capability.tool not in catalog or capability.tool not in semantics:
                if contractions is not None:
                    contractions.append(f"unknown_capability_removed:{capability.tool}")
                continue
            semantic = semantics[capability.tool]
            # The controller-owned manifest is authoritative. The planner is not
            # required to emit effect at all; any emitted value is ignored.
            if capability.effect is None:
                if contractions is not None:
                    contractions.append(f"effect_injected:{capability.tool}:{semantic.effect}")
            elif capability.effect != semantic.effect:
                if contractions is not None:
                    contractions.append(
                        f"effect_overridden:{capability.tool}:"
                        f"{capability.effect}->{semantic.effect}"
                    )
            capability.effect = semantic.effect
            capability.operation = semantic.operation
            capability.reusable = capability.effect in READ_EFFECTS
            capability.origin = "planner"
            if capability.max_calls < 1:
                if contractions is not None:
                    contractions.append(f"nonpositive_capability_removed:{capability.tool}")
                continue
            if capability.effect in SIDE_EFFECTS:
                grounded = SemanticPolicyCompiler._controller_ground_side_effect(
                    query, capability, semantic,
                )
                if grounded is None:
                    if contractions is not None:
                        contractions.append(
                            f"ungrounded_side_effect_removed:{capability.tool}"
                        )
                    continue
                start, end, fragment = grounded
                capability.evidence_start = start
                capability.evidence_end = end
                capability.evidence = fragment
                if contractions is not None:
                    contractions.append(f"side_effect_evidence_grounded:{capability.tool}")
                # An authenticated request such as "email each person" grants a
                # repeated operation, although the collection cardinality is not
                # known until an authorized read executes. Keep this bounded and
                # only widen capabilities whose target comes from a declared
                # source edge; literal destinations remain single-use unless the
                # planner emitted distinct call sites.
                plural_request = bool(re.search(r"\b(each|every|all)\b", query, re.I))
                has_derived_target = any(
                    binding.binding == "derived"
                    and semantic.parameters.get(name) is not None
                    and semantic.parameters[name].authority_sensitive
                    for name, binding in capability.parameters.items()
                )
                if plural_request and has_derived_target:
                    bounded = plural_side_effect_budget
                    if bounded != capability.max_calls and contractions is not None:
                        contractions.append(
                            f"plural_side_effect_budget:{capability.tool}:"
                            f"{capability.max_calls}->{bounded}"
                        )
                    capability.max_calls = bounded
                elif capability.max_calls != 1:
                    if contractions is not None:
                        contractions.append(
                            f"side_effect_budget_clamped:{capability.tool}:"
                            f"{capability.max_calls}->1"
                        )
                    capability.max_calls = 1
            properties = catalog[capability.tool]["parameters"].get("properties", {})
            unknown = set(capability.parameters) - set(properties)
            if unknown:
                if contractions is not None:
                    contractions.extend(
                        f"unknown_binding_removed:{capability.tool}.{name}"
                        for name in sorted(unknown)
                    )
                capability.parameters = {
                    name: binding for name, binding in capability.parameters.items()
                    if name in properties
                }
            # v4 controller-side binding recovery. Some hosted planners
            # (observed on qwen-max) emit the parameter block as a bare string,
            # losing every binding and making the capability unusable. Rebuild
            # deterministic bindings from controller-owned schema and the
            # authenticated request: non-authority-sensitive parameters become
            # free; authority-sensitive parameters are bound only to a value
            # that occurs verbatim and unambiguously in the authenticated text.
            if not capability.parameters and properties:
                recovered = SemanticPolicyCompiler._recover_bindings(
                    query, capability.tool, semantic, properties, contractions,
                )
                if recovered:
                    capability.parameters = recovered
            for name, binding in capability.parameters.items():
                parameter = semantic.parameters.get(name)
                parameter_schema = properties.get(name, {})
                # Normalize a scalar authenticated literal to the controller-owned
                # array schema. This removes serialization-only mismatches such as
                # ``wife@example.com`` versus ["wife@example.com"] without
                # changing the authorized recipient.
                if (
                    parameter_schema.get("type") == "array"
                    and binding.binding in {"literal", "normalized_literal"}
                    and binding.value is not None
                    and not isinstance(binding.value, (list, tuple))
                ):
                    binding.value = [binding.value]
                    if contractions is not None:
                        contractions.append(
                            f"schema_array_literal_normalized:{capability.tool}.{name}"
                        )

                # User-authenticated strings containing placeholders are runtime
                # presentation templates, not exact final literals.  Only
                # non-authority-sensitive parameters may be upgraded implicitly.
                # Sensitive template substitutions require explicit source edges.
                if binding.binding == "literal" and isinstance(binding.value, str):
                    templated = SemanticPolicyCompiler._template_grounding(query, binding.value)
                    if templated is not None and re.search(r"\{[^{}]+\}", binding.value):
                        if parameter is not None and parameter.authority_sensitive:
                            if not binding.source_tools:
                                raise ValueError(
                                    f"Sensitive template lacks sources: {capability.tool}.{name}"
                                )
                        start, end, fragment = templated
                        binding.binding = "template"
                        binding.evidence_start = start
                        binding.evidence_end = end
                        binding.evidence = fragment
                        if contractions is not None:
                            contractions.append(
                                f"authenticated_template_upgraded:{capability.tool}.{name}"
                            )

                if (
                    parameter_schema.get("type") == "boolean"
                    and isinstance(binding.value, str)
                    and binding.value.casefold() in {"true", "false"}
                ):
                    binding.value = binding.value.casefold() == "true"
                    if contractions is not None:
                        contractions.append(
                            f"boolean_literal_normalized:{capability.tool}.{name}"
                        )
                enum = parameter_schema.get("enum") or []
                if (
                    enum
                    and binding.binding in {"literal", "normalized_literal"}
                    and binding.value not in enum
                ):
                    normalized_enum = SemanticPolicyCompiler._permission_enum_alias(
                        query, binding.value, enum, name, parameter,
                    )
                    if normalized_enum is None:
                        raise ValueError(
                            f"Literal is outside controller-owned enum: "
                            f"{capability.tool}.{name}"
                        )
                    enum_value, start, end, fragment = normalized_enum
                    binding.binding = "normalized_literal"
                    binding.value = enum_value
                    binding.evidence_start = start
                    binding.evidence_end = end
                    binding.evidence = fragment
                    binding.inferred_components = ["enum_alias"]
                    if contractions is not None:
                        contractions.append(
                            f"permission_enum_normalized:{capability.tool}.{name}"
                        )
                if binding.binding in {"literal", "normalized_literal", "template"} and binding.value is None:
                    raise ValueError(
                        f"{binding.binding} binding lacks value: {capability.tool}.{name}"
                    )
                if binding.binding == "literal" and not SemanticPolicyCompiler._literal_supported(
                    query,
                    binding.value,
                    parameter_name=name,
                    semantic_role=(parameter.semantic_role if parameter else ""),
                ):
                    grounded = SemanticPolicyCompiler._normalized_literal_grounding(
                        query, binding.value,
                    )
                    if grounded is None:
                        raise ValueError(
                            f"Literal binding is not grounded in authenticated request: "
                            f"{capability.tool}.{name}; use free for a non-sensitive "
                            "operational choice"
                        )
                    start, end, fragment, inferred = grounded
                    binding.binding = "normalized_literal"
                    binding.evidence_start = start
                    binding.evidence_end = end
                    binding.evidence = fragment
                    binding.inferred_components = inferred
                    if contractions is not None:
                        contractions.append(
                            f"authenticated_literal_normalized:{capability.tool}.{name}"
                        )
                if binding.binding == "normalized_literal":
                    if "enum_alias" in binding.inferred_components:
                        inferred = list(binding.inferred_components)
                    else:
                        grounded = SemanticPolicyCompiler._normalized_literal_grounding(
                            query, binding.value,
                        )
                        if grounded is None:
                            raise ValueError(
                                f"Normalized literal is not grounded in authenticated request: "
                                f"{capability.tool}.{name}"
                            )
                        start, end, fragment, inferred = grounded
                        binding.evidence_start = start
                        binding.evidence_end = end
                        binding.evidence = fragment
                        binding.inferred_components = inferred
                    if "inferred_year" in inferred:
                        raise ValueError(
                            f"Temporal binding invents a missing year: "
                            f"{capability.tool}.{name}; derive it from an authorized "
                            "time/context read or omit the optional filter"
                        )
                if binding.binding == "template":
                    grounded_template = SemanticPolicyCompiler._template_grounding(
                        query, binding.value,
                    )
                    if grounded_template is None:
                        # The planner sometimes calls generated/summarized content
                        # a template even though the exact template is not present
                        # in the authenticated request. For a non-sensitive sink,
                        # keep the declared provenance edge and contract the
                        # representation to computed/generated rather than failing
                        # the whole policy compilation. Sensitive templates remain
                        # strictly fail-closed.
                        temporal_parameter = SemanticPolicyCompiler._is_temporal_parameter(
                            name, parameter,
                        )
                        if temporal_parameter:
                            raise ValueError(
                                f"Ungrounded temporal template: {capability.tool}.{name}; "
                                "derive it from an authorized time/context read or omit it"
                            )
                        if parameter is not None and not parameter.authority_sensitive:
                            if binding.source_tools:
                                binding.binding = "computed"
                                binding.value = None
                                binding.allowed_fields = []
                                if contractions is not None:
                                    contractions.append(
                                        f"ungrounded_template_to_computed:"
                                        f"{capability.tool}.{name}"
                                    )
                            else:
                                binding.binding = "generated"
                                binding.value = None
                                binding.allowed_fields = []
                                if contractions is not None:
                                    contractions.append(
                                        f"ungrounded_template_to_generated:"
                                        f"{capability.tool}.{name}"
                                    )
                        else:
                            raise ValueError(
                                f"Template is not grounded in authenticated request: "
                                f"{capability.tool}.{name}"
                            )
                    else:
                        start, end, fragment = grounded_template
                        binding.evidence_start = start
                        binding.evidence_end = end
                        binding.evidence = fragment
                        if (
                            parameter is not None
                            and parameter.authority_sensitive
                            and not binding.source_tools
                        ):
                            raise ValueError(
                                f"Sensitive template lacks sources: "
                                f"{capability.tool}.{name}"
                            )
                if (
                    binding.binding == "derived"
                    and binding.source_tools
                    and parameter is not None
                    and not parameter.authority_sensitive
                    and SemanticPolicyCompiler._query_requests_semantic_transform(query)
                ):
                    # Exact-copy provenance is too strict when the authenticated
                    # user explicitly requests summarization, feedback, or other
                    # semantic transformation. Preserve the same authorized source
                    # edge but mark the non-sensitive sink as computed.
                    binding.binding = "computed"
                    binding.allowed_fields = []
                    if contractions is not None:
                        contractions.append(
                            f"semantic_transform_to_computed:{capability.tool}.{name}"
                        )
                if binding.binding in {"derived", "computed"} and not binding.source_tools:
                    raise ValueError(f"Derived binding lacks sources: {capability.tool}.{name}")
                if (strict_fields and binding.binding == "derived"
                        and (len(binding.source_tools) != 1 or len(binding.allowed_fields) != 1)):
                    raise ValueError(
                        f"Derived binding requires one declared source field: "
                        f"{capability.tool}.{name}"
                    )
                if binding.binding in {"generated", "free"} and parameter and parameter.authority_sensitive:
                    raise ValueError(
                        f"Sensitive parameter cannot be {binding.binding}: {capability.tool}.{name}"
                    )
                if (
                    binding.binding == "generated"
                    and SemanticPolicyCompiler._is_temporal_parameter(name, parameter)
                ):
                    raise ValueError(
                        f"Temporal parameter cannot be generated: {capability.tool}.{name}"
                    )
                if binding.binding not in {"derived", "computed", "template"} and binding.source_tools:
                    binding.source_tools = []
                    binding.allowed_fields = []
                    if contractions is not None:
                        contractions.append(
                            f"irrelevant_sources_removed:{capability.tool}.{name}"
                        )
            accepted.append(capability)
        # A generated/free filename is evidence that the request did not provide
        # an exact source filename. Refine that read to a controller-known content
        # search over the same resource, and preserve downstream source edges.
        # This changes execution strategy only; it creates no side-effect authority.
        replacements: Dict[str, str] = {}
        for index, capability in enumerate(list(accepted)):
            semantic = semantics.get(capability.tool)
            if semantic is None or semantic.effect not in READ_EFFECTS:
                continue
            filename_bindings = [
                capability.parameters[name]
                for name, item in semantic.parameters.items()
                if name in capability.parameters
                and "file" in item.semantic_role.casefold()
                and "name" in item.semantic_role.casefold()
            ]
            if not filename_bindings or not all(
                item.binding in {"generated", "free"} for item in filename_bindings
            ):
                continue
            alternative = None
            for candidate_name, candidate_semantic in semantics.items():
                if (
                    candidate_name == capability.tool
                    or candidate_semantic.effect not in READ_EFFECTS
                    or candidate_semantic.resource != semantic.resource
                ):
                    continue
                roles = " ".join(
                    item.semantic_role.casefold()
                    for item in candidate_semantic.parameters.values()
                )
                if "content" not in roles or "query" not in roles:
                    continue
                alternative = SemanticPolicyCompiler._synthesize_read_dependency(
                    query, candidate_name, candidate_semantic, catalog,
                )
                if alternative is not None:
                    break
            if alternative is None:
                continue
            accepted[index] = alternative
            replacements[capability.tool] = alternative.tool
            if contractions is not None:
                contractions.append(
                    f"ungrounded_filename_read_refined:{capability.tool}->{alternative.tool}"
                )
        if replacements:
            for capability in accepted:
                for binding in capability.parameters.values():
                    binding.source_tools = [
                        replacements.get(source, source) for source in binding.source_tools
                    ]

        # Complete missing READ dependencies before contracting the graph.  This
        # is the key "freeze authority, not execution" step: the planner must
        # declare the data-flow edge, but it need not redundantly spell out every
        # intermediate read node.  The controller synthesizes only read/local
        # compute sources whose required parameters can be filled without
        # inventing sensitive authority.
        synthesized = True
        while synthesized:
            synthesized = False
            allowed = {item.tool for item in accepted}
            missing_sources: List[str] = []
            for capability in accepted:
                for binding in capability.parameters.values():
                    if binding.binding in {"derived", "computed", "template"}:
                        for source in binding.source_tools:
                            if source not in allowed and source != capability.tool:
                                missing_sources.append(source)
            for source in dict.fromkeys(missing_sources):
                semantic = semantics.get(source)
                if semantic is None:
                    continue
                completed = SemanticPolicyCompiler._synthesize_read_dependency(
                    query, source, semantic, catalog,
                )
                if completed is None:
                    continue
                accepted.append(completed)
                allowed.add(source)
                synthesized = True
                if contractions is not None:
                    contractions.append(f"read_dependency_completed:{source}")

        # Compute the greatest dependency-closed subgraph after safe completion.
        # Any still-open edge requires sensitive authority the controller could
        # not derive and therefore remains fail-closed.
        changed = True
        while changed:
            changed = False
            allowed = {item.tool for item in accepted}
            retained = []
            for capability in accepted:
                invalid_sources = set()
                for binding in capability.parameters.values():
                    if binding.binding in {"derived", "computed", "template"}:
                        invalid_sources.update(
                            source for source in binding.source_tools
                            if source not in allowed or source == capability.tool
                        )
                if invalid_sources:
                    if contractions is not None:
                        contractions.append(
                            f"dependency_open_capability_removed:{capability.tool}:"
                            + ",".join(sorted(invalid_sources))
                        )
                    changed = True
                else:
                    retained.append(capability)
            accepted = retained

        # Present source capabilities before their consumers.  This does not
        # change authority; it only makes the already-approved data-flow graph
        # executable by an agent that follows the controller plan left-to-right.
        ordered: List[Capability] = []
        remaining = list(accepted)
        ready_tools: set[str] = set()
        while remaining:
            progressed = False
            for capability in list(remaining):
                dependencies = set()
                for binding in capability.parameters.values():
                    if binding.binding in {"derived", "computed", "template"}:
                        dependencies.update(binding.source_tools)
                if dependencies.issubset(ready_tools):
                    ordered.append(capability)
                    ready_tools.add(capability.tool)
                    remaining.remove(capability)
                    progressed = True
            if not progressed:
                # Cycles should already have been contracted. Preserve order as a
                # fail-safe rather than silently inventing a dependency edge.
                ordered.extend(remaining)
                break
        if ordered != accepted and contractions is not None:
            contractions.append("capability_order_normalized")
        return AuthorizationPlan(goal=plan.goal, capabilities=ordered)

    @staticmethod
    def _parameter_rules(tool: Mapping[str, Any]) -> Dict[str, ParameterRule]:
        schema = tool["parameters"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        rules = {}
        for name, raw in properties.items():
            raw = dict(raw)
            rules[name] = ParameterRule(
                name=name,
                type=raw.get("type", "string"),
                pattern=raw.get("pattern"),
                allowed_values=raw.get("enum"),
                min_length=raw.get("minLength"),
                max_length=raw.get("maxLength"),
                minimum=raw.get("minimum"),
                maximum=raw.get("maximum"),
                required=name in required,
                risk="low",
                provenance=[],
                secret_policy="allow",
            )
        return rules

    def compile(
        self,
        *,
        query: str,
        session_id: str,
        available_tools: Sequence[Mapping[str, Any]],
        ttl: int = 300,
    ) -> SemanticCompileResult:
        tools = [self._normalise_tool(tool) for tool in available_tools]
        manifest = self.compile_manifest(tools)
        contractions: List[str] = []
        def validate_plan(payload: Mapping[str, Any]) -> AuthorizationPlan:
            plan_catalog = {tool["name"]: tool for tool in tools}
            payload = self._repair_plan_payload(payload, contractions, plan_catalog)
            proposed = _model_validate(AuthorizationPlan, payload)
            attempt_contractions: List[str] = []
            validated = self._validate_plan(
                query, proposed, manifest, tools, attempt_contractions,
                self.plural_side_effect_budget,
                strict_fields=self.pinned_manifest is not None,
            )
            contractions[:] = attempt_contractions
            return validated

        plan = self._generate_validated(
            label="authorization_plan",
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_prompt=authorization_plan_prompt(query, _model_dump(manifest), tools),
            validator=validate_plan,
        )
        completeness_second_pass_used = False
        if self.completeness_second_pass:
            semantics_probe = manifest.by_name()
            plan_has_side_effect = any(
                (
                    (semantics_probe[cap.tool].effect
                     if cap.tool in semantics_probe else cap.effect)
                    not in READ_EFFECTS
                )
                for cap in plan.capabilities
            )
            normalized_query = query.casefold()
            multi_step = bool(
                re.search(r"(?:\bthen\b|\band\b|\bafter\b|\bbefore\b|\bfirst\b|\bfinally\b)", normalized_query)
                or re.search(r"(?:^|\s)\d+[.)]", query)
            )
            needs_side_effect_recheck = (
                not plan_has_side_effect and SIDE_EFFECT_CUE_PATTERN.search(query)
            )
            if needs_side_effect_recheck or multi_step:
                payload = json.loads(
                    authorization_plan_prompt(query, _model_dump(manifest), tools)
                )
                feedback = []
                if needs_side_effect_recheck:
                    feedback.append(PLAN_COMPLETENESS_FEEDBACK)
                if multi_step:
                    feedback.append(MULTI_STEP_COMPLETENESS_FEEDBACK)
                if re.search(
                    r"\b(?:pay|refund|reimburse|transfer|send\s+(?:them\s+)?the\s+difference)\b",
                    normalized_query,
                ):
                    feedback.append(FINANCIAL_COMPLETENESS_FEEDBACK)
                payload["controller_feedback"] = " ".join(feedback)
                payload["previous_validated_plan"] = _model_dump(plan)
                try:
                    candidate = self._generate_validated(
                        label="authorization_plan_completeness",
                        system_prompt=PLAN_SYSTEM_PROMPT,
                        user_prompt=json.dumps(
                            payload, ensure_ascii=False, sort_keys=True,
                        ),
                        validator=validate_plan,
                    )
                    def coverage(item: AuthorizationPlan) -> tuple[int, int, int]:
                        effects = [
                            semantics_probe[cap.tool].effect
                            if cap.tool in semantics_probe else cap.effect
                            for cap in item.capabilities
                        ]
                        return (
                            sum(effect not in READ_EFFECTS for effect in effects),
                            len({cap.tool for cap in item.capabilities}),
                            len(item.capabilities),
                        )
                    # A completeness pass may add authority only when every
                    # side effect remains independently grounded by validation.
                    # Never replace a more complete first plan with a smaller one.
                    if coverage(candidate) >= coverage(plan):
                        plan = candidate
                        completeness_second_pass_used = True
                except ValueError:
                    # Keep the first validated plan; the re-check is best-effort
                    # and must never weaken the fail-closed boundary.
                    pass
        plan_by_tool = plan.by_tool_all()
        semantics = manifest.by_name()
        blueprint_tools = {}
        read_budget_total = 0
        side_effect_budget_total = 0
        read_defaults_issued = []
        for tool in tools:
            name = tool["name"]
            capabilities = plan_by_tool.get(name, [])
            semantic = semantics.get(name)
            if (
                not capabilities
                and self.read_tools_open
                and semantic is not None
                and semantic.effect in READ_EFFECTS
            ):
                # v3 read-open default: the controller-owned manifest classifies
                # this tool as read/local-compute. Issue a reusable, free-bound
                # capability so the agent is not silently starved of reads the
                # planner did not foresee. Side-effect tools stay plan-bound.
                read_capability = Capability(
                    tool=name,
                    effect=semantic.effect,
                    operation=semantic.operation,
                    reusable=True,
                    origin="controller_read_default",
                    max_calls=self.read_tool_budget,
                    parameters={
                        param_name: ParameterBinding(binding="free")
                        for param_name in semantic.parameters
                    },
                )
                plan.capabilities.append(read_capability)
                capabilities = [read_capability]
                read_defaults_issued.append(name)
            if capabilities and semantic is not None and semantic.effect in READ_EFFECTS:
                # Read authority is reusable: the plan freezes which resource and
                # data-flow edges are legal, while a separate controller resource
                # budget prevents unbounded loops.
                tool_budget = max(
                    self.read_tool_budget,
                    sum(max(1, int(cap.max_calls)) for cap in capabilities),
                )
                read_budget_total += tool_budget
            elif capabilities:
                tool_budget = sum(max(1, int(cap.max_calls)) for cap in capabilities)
                side_effect_budget_total += tool_budget
            else:
                tool_budget = 1
            blueprint_tools[name] = ToolBlueprint(
                tool_id=name,
                permission=(ToolPermission.ALLOWED if capabilities else ToolPermission.DENIED),
                description=tool["description"],
                parameter_rules=self._parameter_rules(tool) if capabilities else {},
                max_calls_per_session=tool_budget,
                requires_confirmation=False,
                secrets="allow",
            )
        total_calls = (
            min(self.session_read_budget, read_budget_total)
            + side_effect_budget_total
        )
        blueprint = SessionBlueprint(
            session_id=session_id,
            user_intent=query,
            authorized_goals=[plan.goal],
            allowed_tools=blueprint_tools,
            forbidden_patterns=[],
            ttl_seconds=ttl,
            max_total_tool_calls=max(1, total_calls),
            metadata={
                "defense": "toolfence_semantic",
                "compiler": "isolated_same_model_json",
                "plan": _model_dump(plan),
                "manifest": _model_dump(manifest),
                "compiler_contractions": list(contractions),
                "read_tool_budget": self.read_tool_budget,
                "session_read_budget": self.session_read_budget,
                "authority_model": "freeze_authority_not_execution",
                "read_tools_open": self.read_tools_open,
                "read_defaults_issued": read_defaults_issued,
                "completeness_second_pass": self.completeness_second_pass,
                "completeness_second_pass_used": completeness_second_pass_used,
            },
        )
        return SemanticCompileResult(
            blueprint=blueprint,
            manifest=manifest,
            plan=plan,
            compiler_contractions=tuple(contractions),
        )
