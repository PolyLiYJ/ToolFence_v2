"""Formal authorization primitives for the ICLR 2027 ToolFence redesign.

This module implements the algorithmic core independently from any benchmark:
provenance-aware compilation (Alg. 1), stateful expansion detection and
monotonic updates (Alg. 2/5), a pre-execution authorization gate (Alg. 3's
runtime invariant), and minimum-change action repair (Alg. 4).
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from .blueprint import SessionBlueprint, ToolBlueprint, ToolPermission


class Provenance(str, Enum):
    TRUSTED_USER = "trusted_user"
    TRUSTED_POLICY = "trusted_policy"
    USER_LITERAL = "user_literal"
    TRUSTED_CONTACTS = "trusted_contacts"
    CONTROLLER_SECRET = "controller_secret"
    DELEGATED = "delegated"
    DECLASSIFIED = "declassified"
    UNTRUSTED_TOOL = "untrusted_tool"
    UNTRUSTED_WEB = "untrusted_web"
    UNTRUSTED_DOCUMENT = "untrusted_document"
    UNSIGNED_PERSISTENT = "unsigned_persistent"
    UNKNOWN = "unknown"

    @property
    def trusted(self) -> bool:
        return self in {
            self.TRUSTED_USER, self.TRUSTED_POLICY, self.USER_LITERAL,
            self.TRUSTED_CONTACTS, self.CONTROLLER_SECRET,
        }


@dataclass(frozen=True)
class ArgumentSource:
    """Controller-side evidence for the origin of one argument value."""

    provenance: Provenance
    source: str = ""
    authenticated: bool = False
    tainted: bool = False
    hops: int = 0
    manifest_id: str = ""

    @property
    def policy_name(self) -> str:
        if self.provenance == Provenance.DELEGATED and self.source:
            return f"delegated:{self.source}".lower()
        if self.provenance == Provenance.DECLASSIFIED and self.source:
            return f"declassified:{self.source}".lower()
        return self.provenance.value.lower()


@dataclass(frozen=True)
class ProvenanceEnvelope:
    content: Any
    provenance: Provenance
    source: str = ""
    authenticated: bool = False
    argument_sources: Mapping[str, ArgumentSource] = field(default_factory=dict)
    tainted_context: bool = False

    @property
    def may_expand_authority(self) -> bool:
        return self.provenance.trusted and self.authenticated


@dataclass(frozen=True)
class ProvenanceManifest:
    """Signed label that survives persistent storage and agent boundaries."""

    artifact_id: str
    content_hash: str
    provenance: Provenance
    source: str
    session_id: str
    agent_id: str = ""
    tainted: bool = True
    hops: int = 0
    parent_ids: Tuple[str, ...] = ()
    issued_at: float = 0.0
    signature: str = ""

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "provenance": self.provenance.value,
            "source": self.source,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tainted": self.tainted,
            "hops": self.hops,
            "parent_ids": list(self.parent_ids),
            "issued_at": self.issued_at,
        }


class ManifestSigner:
    """HMAC authority for persistent provenance manifests.

    Only the controller should possess ``key``. Agents transport manifests but
    cannot mint a trusted label or remove taint without invalidating it.
    """

    def __init__(self, key: str | bytes):
        if not key:
            raise ValueError("A non-empty manifest signing key is required")
        self._key = key.encode() if isinstance(key, str) else key

    @staticmethod
    def content_hash(content: Any) -> str:
        if not isinstance(content, (str, bytes)):
            content = json.dumps(content, sort_keys=True, separators=(",", ":"))
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()

    def _signature(self, fields: Mapping[str, Any]) -> str:
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def issue(
        self,
        content: Any,
        *,
        artifact_id: str,
        provenance: Provenance,
        source: str,
        session_id: str,
        agent_id: str = "",
        tainted: Optional[bool] = None,
        hops: int = 0,
        parent_ids: Sequence[str] = (),
    ) -> ProvenanceManifest:
        if tainted is None:
            tainted = not provenance.trusted
        manifest = ProvenanceManifest(
            artifact_id=artifact_id,
            content_hash=self.content_hash(content),
            provenance=provenance,
            source=source,
            session_id=session_id,
            agent_id=agent_id,
            tainted=tainted,
            hops=hops,
            parent_ids=tuple(parent_ids),
            issued_at=time.time(),
        )
        return replace(manifest, signature=self._signature(manifest.unsigned_dict()))

    def verify(self, content: Any, manifest: ProvenanceManifest) -> bool:
        return (
            hmac.compare_digest(manifest.content_hash, self.content_hash(content))
            and hmac.compare_digest(manifest.signature, self._signature(manifest.unsigned_dict()))
        )

    def source_for(self, content: Any, manifest: Optional[ProvenanceManifest]) -> ArgumentSource:
        if manifest is None or not self.verify(content, manifest):
            return ArgumentSource(Provenance.UNSIGNED_PERSISTENT, tainted=True)
        return ArgumentSource(
            manifest.provenance,
            source=manifest.source,
            authenticated=True,
            tainted=manifest.tainted,
            hops=manifest.hops + 1,
            manifest_id=manifest.artifact_id,
        )


@dataclass(frozen=True)
class SourceValue:
    value: str
    source: ArgumentSource


class SourceRegistry:
    """Small controller-owned registry used for conservative value attribution."""

    def __init__(self):
        self._values: List[SourceValue] = []

    def register(self, value: Any, source: ArgumentSource) -> None:
        if value is None:
            return
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, default=str)
        )
        if rendered:
            self._values.append(SourceValue(rendered, source))

    def register_user_query(self, query: str, source: str = "user_query") -> None:
        self.register(query, ArgumentSource(Provenance.USER_LITERAL, source, authenticated=True))

    def register_observation(self, value: Any, provenance: Provenance, source: str, hops: int = 0) -> None:
        self.register(value, ArgumentSource(provenance, source, tainted=True, hops=hops))

    def register_controller_secret(self, value: str, secret_id: str) -> None:
        self.register(value, ArgumentSource(
            Provenance.CONTROLLER_SECRET, secret_id, authenticated=True, tainted=False,
        ))

    @staticmethod
    def _normalise(value: Any) -> str:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, default=str)
        )
        return " ".join(text.casefold().split())

    def attribute(self, value: Any, fuzzy_threshold: float = 0.94) -> Optional[ArgumentSource]:
        needle = self._normalise(value)
        if not needle:
            return None
        exact: List[ArgumentSource] = []
        fuzzy: List[Tuple[float, ArgumentSource]] = []
        for entry in self._values:
            haystack = self._normalise(entry.value)
            if needle == haystack or needle in haystack:
                exact.append(entry.source)
            elif len(needle) >= 5:
                score = SequenceMatcher(None, needle, haystack).ratio()
                if score >= fuzzy_threshold:
                    fuzzy.append((score, entry.source))
        candidates = exact or [source for _, source in sorted(fuzzy, key=lambda item: item[0], reverse=True)]
        if not candidates:
            return None
        # If the same value occurs in trusted and tainted inputs, trusted literal
        # authority is sufficient only when it was independently present in the
        # authenticated user request.
        user_literal = next((s for s in candidates if s.provenance == Provenance.USER_LITERAL), None)
        return user_literal or candidates[0]

    def declassify(self, value: Any, kind: str, allowed_types: Sequence[str]) -> ArgumentSource:
        """Open only an explicitly configured low-capacity typed channel."""
        kind = kind.lower()
        if kind not in {item.lower() for item in allowed_types}:
            raise ValueError(f"Declassification type '{kind}' is not allowed")
        if kind == "boolean" and not isinstance(value, bool):
            raise ValueError("Boolean declassification requires a bool")
        if kind.startswith("enum:"):
            choices = kind.partition(":")[2].split("|")
            if str(value).lower() not in choices or len(choices) > 16:
                raise ValueError("Enum declassification must use one of at most 16 declared values")
        source = ArgumentSource(Provenance.DECLASSIFIED, kind, authenticated=True, tainted=False)
        self.register(value, source)
        return source


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = json.loads(value)
        tool = value.get("tool") or value.get("name") or value.get("action")
        if not tool:
            raise ValueError("Action must contain tool/name/action")
        args = value.get("arguments", value.get("params", {}))
        if not isinstance(args, Mapping):
            raise ValueError("Action arguments must be an object")
        return cls(str(tool), dict(args))

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass
class AuthorizationDecision:
    allowed: bool
    violations: List[str] = field(default_factory=list)
    authority_distance: float = 0.0
    impact: float = 0.0

    @property
    def risk(self) -> float:
        return self.authority_distance * self.impact


@dataclass
class CapabilityLedger:
    """Trajectory-level state used to stop capability growth by composition."""

    total_calls: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    contacted_hosts: Set[str] = field(default_factory=set)
    observed_secret_hashes: Set[str] = field(default_factory=set)
    permission_levels: Set[str] = field(default_factory=set)
    shared_writes: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _secret_hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def observe_secret(self, value: str, source: str) -> None:
        digest = self._secret_hash(value)
        self.observed_secret_hashes.add(digest)
        self.events.append({"type": "secret_observed", "source": source, "secret_hash": digest})

    def is_observed_secret(self, value: Any) -> bool:
        return isinstance(value, str) and self._secret_hash(value) in self.observed_secret_hashes

    @staticmethod
    def action_hosts(action: Action) -> Set[str]:
        hosts: Set[str] = set()
        for key in ("url", "endpoint", "host"):
            if key not in action.arguments:
                continue
            raw = str(action.arguments[key])
            host = (urlparse(raw).hostname if "://" in raw else raw.split("/")[0]).lower()
            if host:
                hosts.add(host)
        return hosts

    @staticmethod
    def is_shared_write(action: Action) -> bool:
        lowered = action.tool.lower()
        return any(word in lowered for word in (
            "write", "post", "send", "publish", "create", "update",
            "delete", "transfer", "upload", "store", "append",
        ))

    def record_dispatch(self, action: Action) -> None:
        self.total_calls += 1
        self.tool_calls[action.tool] = self.tool_calls.get(action.tool, 0) + 1
        self.contacted_hosts.update(self.action_hosts(action))
        if self.is_shared_write(action):
            self.shared_writes.append({"tool": action.tool, "arguments": dict(action.arguments)})
        self.events.append({"type": "dispatch", "action": action.to_dict()})

    def budget_violations(self, action: Action, blueprint: SessionBlueprint) -> List[str]:
        violations: List[str] = []
        if self.total_calls >= blueprint.max_total_tool_calls:
            violations.append("budget:session_calls:exhausted")
        spec = blueprint.allowed_tools.get(action.tool)
        if spec and self.tool_calls.get(action.tool, 0) >= spec.max_calls_per_session:
            violations.append(f"budget:{action.tool}:exhausted")

        limits = getattr(blueprint, "global_constraints", {})
        for key, current, increments in (
            (
                "max_unique_hosts",
                len(self.contacted_hosts),
                bool(self.action_hosts(action) - self.contacted_hosts),
            ),
            (
                "max_shared_writes",
                len(self.shared_writes),
                self.is_shared_write(action),
            ),
        ):
            if key not in limits:
                continue
            try:
                limit = int(limits[key])
            except (TypeError, ValueError):
                violations.append(f"policy:{key}:invalid")
                continue
            if limit < 0:
                violations.append(f"policy:{key}:invalid")
            elif increments and current >= limit:
                violations.append(f"budget:{key}:exhausted")
        return violations


_SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|bearer)\s*[:=]?\s*"
    r"([A-Za-z0-9_\-./+=]{8,})|\b(?:sk|ghp|hf)_[A-Za-z0-9_\-]{12,}\b"
)


def _looks_like_secret(name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if any(token in name.lower() for token in ("secret", "token", "password", "credential", "api_key")):
        return True
    return bool(_SECRET_PATTERN.search(value))


def _host_allowed(host: str, scopes: Sequence[str]) -> bool:
    host = host.rstrip(".").lower()
    for scope in scopes:
        scope = scope.rstrip(".").lower()
        if scope.startswith("*.") and host.endswith(scope[1:]) and host != scope[2:]:
            return True
        if host == scope:
            return True
    return False


def _source_allowed(source: ArgumentSource, policies: Sequence[str]) -> bool:
    actual = source.policy_name
    normalized = {str(policy).strip().lower() for policy in policies}
    if actual in normalized:
        return True
    if actual.startswith("delegated:") and "delegated" in normalized:
        return True
    if actual.startswith("declassified:") and "declassified" in normalized:
        return True
    # Legacy blueprints used TRUSTED_USER where the method now uses USER_LITERAL.
    return (
        source.provenance in {Provenance.TRUSTED_USER, Provenance.USER_LITERAL}
        and bool(normalized & {"trusted_user", "user_literal"})
    )


def _permission_atoms(blueprint: SessionBlueprint) -> Set[Tuple[str, str]]:
    atoms: Set[Tuple[str, str]] = set()
    for tool, spec in blueprint.allowed_tools.items():
        if spec.permission == ToolPermission.ALLOWED:
            atoms.add(("tool", tool))
            for name, rule in spec.parameter_rules.items():
                if rule.allowed_values:
                    atoms.update((f"arg:{tool}:{name}", str(v)) for v in rule.allowed_values)
    atoms.update(("data", str(v)) for v in getattr(blueprint, "data_domains", []))
    atoms.update(("recipient", str(v)) for v in getattr(blueprint, "allowed_recipients", []))
    return atoms


def _rule_expands(current: Any, proposed: Any) -> bool:
    """Return True when ``proposed`` admits a value/source current denied."""
    if current.allowed_values is not None:
        if proposed.allowed_values is None:
            return True
        if not set(map(str, proposed.allowed_values)).issubset(set(map(str, current.allowed_values))):
            return True
    if current.forbidden_values:
        if not proposed.forbidden_values or not set(map(str, current.forbidden_values)).issubset(
            set(map(str, proposed.forbidden_values))
        ):
            return True
    if current.host_scope:
        if not proposed.host_scope or not set(proposed.host_scope).issubset(set(current.host_scope)):
            return True
    if current.provenance:
        if not proposed.provenance or not set(map(str.lower, proposed.provenance)).issubset(
            set(map(str.lower, current.provenance))
        ):
            return True
    if current.maximum is not None and (proposed.maximum is None or proposed.maximum > current.maximum):
        return True
    if current.minimum is not None and (proposed.minimum is None or proposed.minimum < current.minimum):
        return True
    if current.max_length is not None and (
        proposed.max_length is None or proposed.max_length > current.max_length
    ):
        return True
    if current.min_length is not None and (
        proposed.min_length is None or proposed.min_length < current.min_length
    ):
        return True
    if current.pattern and proposed.pattern != current.pattern:
        # Regex containment is undecidable in general; require exact equality.
        return True
    if current.required and not proposed.required:
        return True
    secret_policy_rank = {"provisioned_only": 0, "deny_observed": 1, "allow": 2}
    if secret_policy_rank.get(proposed.secret_policy, 2) > secret_policy_rank.get(
        current.secret_policy, 2
    ):
        return True
    if not set(proposed.declassify_types).issubset(set(current.declassify_types)):
        return True
    return False


def _blueprint_expands(current: SessionBlueprint, proposed: SessionBlueprint) -> bool:
    if not set(proposed.authorized_tools).issubset(set(current.authorized_tools)):
        return True
    if proposed.max_total_tool_calls > current.max_total_tool_calls:
        return True
    if current.data_domains and (
        not proposed.data_domains or not set(proposed.data_domains).issubset(set(current.data_domains))
    ):
        return True
    if current.allowed_recipients and (
        not proposed.allowed_recipients
        or not set(proposed.allowed_recipients).issubset(set(current.allowed_recipients))
    ):
        return True
    if not set(current.forbidden_patterns).issubset(set(proposed.forbidden_patterns)):
        return True
    # Known trajectory ceilings are monotone: removing or increasing one is an
    # authority expansion. Unknown metadata is deliberately not interpreted as
    # security policy by this generic core.
    for name in ("max_unique_hosts", "max_shared_writes"):
        if name not in current.global_constraints:
            continue
        if name not in proposed.global_constraints:
            return True
        try:
            if int(proposed.global_constraints[name]) > int(current.global_constraints[name]):
                return True
        except (TypeError, ValueError):
            return True
    for tool in proposed.authorized_tools:
        old_spec = current.allowed_tools[tool]
        new_spec = proposed.allowed_tools[tool]
        if new_spec.max_calls_per_session > old_spec.max_calls_per_session:
            return True
        if old_spec.requires_confirmation and not new_spec.requires_confirmation:
            return True
        egress_rank = {"deny": 0, "scoped": 1, "allow": 2}
        if egress_rank.get(new_spec.egress, 2) > egress_rank.get(old_spec.egress, 2):
            return True
        secret_rank = {"provisioned_only": 0, "deny_observed": 1, "allow": 2}
        if secret_rank.get(new_spec.secrets, 2) > secret_rank.get(old_spec.secrets, 2):
            return True
        for name, old_rule in old_spec.parameter_rules.items():
            if name not in new_spec.parameter_rules and old_rule.required:
                return True
        for name, new_rule in new_spec.parameter_rules.items():
            old_rule = old_spec.parameter_rules.get(name)
            if old_rule is None or _rule_expands(old_rule, new_rule):
                return True
    return False


def _intersect_parameter_rules(rules: Sequence[Any]) -> Any:
    """Compute a conservative, field-wise intersection of compiler proposals."""
    merged = copy.deepcopy(rules[0])
    restricted_values = [set(map(str, r.allowed_values)) for r in rules if r.allowed_values is not None]
    if restricted_values:
        merged.allowed_values = sorted(set.intersection(*restricted_values))
    provenance = [set(map(str.lower, r.provenance)) for r in rules if r.provenance]
    if provenance:
        merged.provenance = sorted(set.intersection(*provenance))
    host_scopes = [set(r.host_scope) for r in rules if r.host_scope]
    if host_scopes:
        merged.host_scope = sorted(set.intersection(*host_scopes))
    minima = [r.minimum for r in rules if r.minimum is not None]
    maxima = [r.maximum for r in rules if r.maximum is not None]
    min_lengths = [r.min_length for r in rules if r.min_length is not None]
    max_lengths = [r.max_length for r in rules if r.max_length is not None]
    merged.minimum = max(minima) if minima else None
    merged.maximum = min(maxima) if maxima else None
    merged.min_length = max(min_lengths) if min_lengths else None
    merged.max_length = min(max_lengths) if max_lengths else None
    merged.required = any(r.required for r in rules)
    policy_rank = {"allow": 0, "deny_observed": 1, "provisioned_only": 2}
    merged.secret_policy = max(
        (r.secret_policy for r in rules), key=lambda value: policy_rank.get(value, 2)
    )
    patterns = list(dict.fromkeys(r.pattern for r in rules if r.pattern))
    if len(patterns) == 1:
        merged.pattern = patterns[0]
    elif patterns:
        merged.pattern = "".join(f"(?={pattern})" for pattern in patterns) + ".*"
    return merged


class ConservativeBlueprintCompiler:
    """Risk-sensitive conservative consensus compiler (Algorithm 1)."""

    def __init__(self, tau0: float = 0.5, risk_lambda: float = 0.5):
        if not 0 <= tau0 <= 1 or risk_lambda < 0:
            raise ValueError("tau0 must be in [0,1] and risk_lambda non-negative")
        self.tau0 = tau0
        self.risk_lambda = risk_lambda

    def compile(
        self,
        candidates: Sequence[SessionBlueprint],
        risk: Optional[Callable[[str], float]] = None,
        session_id: Optional[str] = None,
    ) -> SessionBlueprint:
        if not candidates:
            raise ValueError("At least one candidate blueprint is required")
        risk = risk or (lambda _tool: 0.0)
        support: Dict[str, int] = {}
        specs: Dict[str, List[ToolBlueprint]] = {}
        for candidate in candidates:
            for tool in candidate.authorized_tools:
                support[tool] = support.get(tool, 0) + 1
                specs.setdefault(tool, []).append(candidate.allowed_tools[tool])

        allowed: Dict[str, ToolBlueprint] = {}
        thresholds: Dict[str, float] = {}
        for tool, count in support.items():
            threshold = min(1.0, self.tau0 + self.risk_lambda * max(0.0, min(1.0, risk(tool))))
            thresholds[tool] = threshold
            if count / len(candidates) >= threshold:
                # Parameter constraints are intersected: only unanimously proposed
                # rules survive, preventing one compiler from widening arguments.
                base = copy.deepcopy(specs[tool][0])
                common = set(base.parameter_rules)
                for spec in specs[tool][1:]:
                    common &= set(spec.parameter_rules)
                base.parameter_rules = {
                    name: _intersect_parameter_rules([
                        spec.parameter_rules[name] for spec in specs[tool]
                    ])
                    for name in common
                }
                base.max_calls_per_session = min(spec.max_calls_per_session for spec in specs[tool])
                base.requires_confirmation = any(spec.requires_confirmation for spec in specs[tool])
                base.permission = ToolPermission.ALLOWED
                allowed[tool] = base

        first = candidates[0]
        def restricted_intersection(attribute: str) -> List[str]:
            restricted = [
                set(getattr(item, attribute, []))
                for item in candidates
                if getattr(item, attribute, [])
            ]
            return sorted(set.intersection(*restricted)) if restricted else []

        return SessionBlueprint(
            session_id=session_id or first.session_id,
            user_intent=first.user_intent,
            allowed_tools=allowed,
            authorized_goals=list(getattr(first, "authorized_goals", [])),
            data_domains=restricted_intersection("data_domains"),
            allowed_recipients=restricted_intersection("allowed_recipients"),
            context_constraints=dict(getattr(first, "context_constraints", {})),
            global_constraints=dict(getattr(first, "global_constraints", {})),
            ttl_seconds=min(item.ttl_seconds for item in candidates),
            max_total_tool_calls=min(item.max_total_tool_calls for item in candidates),
            forbidden_patterns=sorted({
                pattern for item in candidates for pattern in item.forbidden_patterns
            }),
            required_confirmations=sorted({
                value for item in candidates for value in item.required_confirmations
            }),
            metadata={
                **first.metadata,
                "compiler": "risk_sensitive_consensus",
                "compiler_count": len(candidates),
                "support": support,
                "thresholds": thresholds,
            },
        )


class BlueprintGrammarCompiler:
    """Compile a blueprint to a backend-neutral JSON Schema (Algorithm 3).

    Outlines, vLLM guided decoding, or another CFG backend can consume this
    schema. The pre-execution gate remains mandatory because a backend may be
    unavailable or incorrectly configured.
    """

    def compile_json_schema(self, blueprint: SessionBlueprint) -> Dict[str, Any]:
        variants: List[Dict[str, Any]] = []
        for tool in blueprint.authorized_tools:
            spec = blueprint.allowed_tools[tool]
            properties: Dict[str, Any] = {}
            required_arguments: List[str] = []
            for name, rule in spec.parameter_rules.items():
                json_type = rule.type if rule.type in {
                    "string", "integer", "number", "boolean", "array", "object"
                } else "string"
                item: Dict[str, Any] = {"type": json_type}
                if rule.allowed_values:
                    item["enum"] = list(rule.allowed_values)
                if rule.pattern:
                    item["pattern"] = rule.pattern
                if rule.min_length is not None:
                    item["minLength"] = rule.min_length
                if rule.max_length is not None:
                    item["maxLength"] = rule.max_length
                if rule.minimum is not None:
                    item["minimum"] = rule.minimum
                if rule.maximum is not None:
                    item["maximum"] = rule.maximum
                if rule.host_scope:
                    item["x-toolfence-host-scope"] = list(rule.host_scope)
                if rule.provenance:
                    item["x-toolfence-provenance"] = list(rule.provenance)
                properties[name] = item
                if rule.required:
                    required_arguments.append(name)
            variants.append({
                "type": "object",
                "properties": {
                    "tool": {"const": tool},
                    "arguments": {
                        "type": "object", "properties": properties,
                        "required": required_arguments,
                        "additionalProperties": False,
                    },
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            })
        # Empty blueprints intentionally describe an uninhabited action space.
        return {"$schema": "https://json-schema.org/draft/2020-12/schema", "oneOf": variants}


class AuthorizationGate:
    """Validates a parsed action immediately before tool execution."""

    DEFAULT_IMPACT = {"read": 0.25, "search": 0.2, "list": 0.25,
                      "write": 0.7, "create": 0.6, "send": 1.0,
                      "delete": 1.0, "execute": 1.0, "transfer": 1.0}

    def __init__(self, impact_overrides: Optional[Mapping[str, float]] = None):
        self.impacts = {**self.DEFAULT_IMPACT, **dict(impact_overrides or {})}

    def impact(self, tool: str) -> float:
        lowered = tool.lower()
        return max((score for prefix, score in self.impacts.items() if prefix in lowered), default=0.5)

    def evaluate(
        self,
        action: Action,
        blueprint: SessionBlueprint,
        envelope: Optional[ProvenanceEnvelope] = None,
        ledger: Optional[CapabilityLedger] = None,
        sources: Optional[SourceRegistry] = None,
    ) -> AuthorizationDecision:
        violations: List[str] = []
        if action.tool not in blueprint.allowed_tools:
            violations.append(f"tool:{action.tool}:not_authorized")
        else:
            ok, reason = blueprint.validate_tool_call(action.tool, dict(action.arguments))
            if not ok:
                violations.append(reason)

            rendered_arguments = json.dumps(
                dict(action.arguments), sort_keys=True, default=str,
            )
            content_safe, matched_pattern = blueprint.check_content_safety(rendered_arguments)
            if not content_safe:
                violations.append(f"content:forbidden_pattern:{matched_pattern}")

            spec = blueprint.allowed_tools[action.tool]
            if spec.requires_confirmation and not (envelope and envelope.authenticated):
                violations.append(f"tool:{action.tool}:confirmation_required")
            if spec.egress == "deny" and any(
                key in action.arguments for key in ("url", "endpoint", "host", "destination")
            ):
                violations.append(f"tool:{action.tool}:egress_denied")

            for name, value in action.arguments.items():
                rule = spec.parameter_rules.get(name)
                if rule is None:
                    continue
                source = envelope.argument_sources.get(name) if envelope else None
                if source is None and sources is not None:
                    source = sources.attribute(value)

                if rule.provenance:
                    if source is None:
                        violations.append(f"argument:{name}:provenance_unknown")
                    elif not _source_allowed(source, rule.provenance):
                        violations.append(
                            f"argument:{name}:provenance_{source.policy_name}_not_allowed"
                        )

                if rule.risk.lower() == "high" and envelope and envelope.tainted_context and source is None:
                    violations.append(f"argument:{name}:confirmation_required_for_tainted_context")

                if rule.host_scope and isinstance(value, str):
                    host = urlparse(value).hostname
                    if not host or not _host_allowed(host, rule.host_scope):
                        violations.append(f"argument:{name}:host_not_in_scope")

                observed_secret = ledger.is_observed_secret(value) if ledger else False
                secret_source_ok = source is not None and source.provenance == Provenance.CONTROLLER_SECRET
                if rule.secret_policy == "provisioned_only" and not secret_source_ok:
                    violations.append(f"argument:{name}:secret_not_controller_provisioned")
                elif (
                    rule.secret_policy != "allow"
                    and (observed_secret or _looks_like_secret(name, value))
                    and not secret_source_ok
                ):
                    violations.append(f"argument:{name}:observed_secret_flow_denied")
                if (
                    spec.secrets == "provisioned_only"
                    and _looks_like_secret(name, value)
                    and not secret_source_ok
                ):
                    violations.append(f"argument:{name}:secret_not_controller_provisioned")

        recipients = set(getattr(blueprint, "allowed_recipients", []))
        destination = action.arguments.get("to") or action.arguments.get("recipient") or action.arguments.get("destination")
        if destination is not None and recipients and str(destination) not in recipients:
            violations.append(f"recipient:{destination}:not_authorized")

        domains = set(getattr(blueprint, "data_domains", []))
        domain = action.arguments.get("data_domain") or action.arguments.get("domain")
        if domain is not None and domains and str(domain) not in domains:
            violations.append(f"data_domain:{domain}:not_authorized")

        if ledger is not None:
            violations.extend(ledger.budget_violations(action, blueprint))

        # Four dimensions correspond to tool, argument/context, data, recipient.
        violations = list(dict.fromkeys(violations))
        distance = min(1.0, len(violations) / 4.0)
        return AuthorizationDecision(not violations, violations, distance, self.impact(action.tool))


class StatefulAuthorityTracker:
    """Session state implementing Algorithms 2 and 5."""

    def __init__(
        self,
        initial_blueprint: SessionBlueprint,
        gate: Optional[AuthorizationGate] = None,
        ledger: Optional[CapabilityLedger] = None,
        sources: Optional[SourceRegistry] = None,
    ):
        self.initial = copy.deepcopy(initial_blueprint)
        self.current = copy.deepcopy(initial_blueprint)
        self.gate = gate or AuthorizationGate()
        self.ledger = ledger or CapabilityLedger()
        self.sources = sources or SourceRegistry()
        self.history: List[Dict[str, Any]] = []

    def inspect_action(self, action: Any, envelope: ProvenanceEnvelope) -> AuthorizationDecision:
        parsed = Action.from_obj(action)
        decision = self.gate.evaluate(parsed, self.current, envelope, self.ledger, self.sources)
        self.history.append({"type": "action", "source": envelope.source,
                             "provenance": envelope.provenance.value,
                             "action": parsed.to_dict(), "decision": decision})
        return decision

    def update(self, proposed: SessionBlueprint, envelope: ProvenanceEnvelope) -> SessionBlueprint:
        current_atoms = _permission_atoms(self.current)
        proposed_atoms = _permission_atoms(proposed)
        expands = _blueprint_expands(self.current, proposed)
        if expands and not envelope.may_expand_authority:
            self.history.append({"type": "rejected_update", "source": envelope.source,
                                 "provenance": envelope.provenance.value,
                                 "new_permissions": sorted(proposed_atoms - current_atoms)})
            return copy.deepcopy(self.current)
        # Trusted authenticated updates may change authority, while untrusted
        # updates are accepted only when they are a restriction.
        self.current = copy.deepcopy(proposed)
        self.history.append({"type": "accepted_update", "source": envelope.source,
                             "provenance": envelope.provenance.value, "expanded": expands})
        return copy.deepcopy(self.current)

    def observe(
        self,
        content: Any,
        provenance: Provenance = Provenance.UNTRUSTED_TOOL,
        source: str = "tool_output",
        hops: int = 0,
    ) -> None:
        """Register an observation and quarantine secret-like values it contains."""
        self.sources.register_observation(content, provenance, source, hops)
        if isinstance(content, str):
            for match in _SECRET_PATTERN.finditer(content):
                secret = match.group(1) or match.group(0)
                self.ledger.observe_secret(secret, source)
        self.history.append({
            "type": "observation", "source": source, "provenance": provenance.value,
            "tainted": not provenance.trusted, "hops": hops,
        })

    def record_dispatch(self, action: Action) -> None:
        self.ledger.record_dispatch(action)


@dataclass
class RepairResult:
    action: Optional[Action]
    repaired: bool
    cost: float
    violations: List[str]
    abstained: bool = False


class MinimumChangeRepairer:
    """Deterministic candidate search for Algorithm 4."""

    def __init__(self, gate: Optional[AuthorizationGate] = None,
                 tool_cost: float = 1.0, argument_cost: float = 0.25,
                 utility_threshold: float = 0.5,
                 tool_substitutions: Optional[Mapping[str, Sequence[str]]] = None):
        self.gate = gate or AuthorizationGate()
        self.tool_cost = tool_cost
        self.argument_cost = argument_cost
        self.utility_threshold = utility_threshold
        self.tool_substitutions = {
            key: tuple(values) for key, values in (tool_substitutions or {}).items()
        }

    def repair(self, action: Any, blueprint: SessionBlueprint,
               utility: Optional[Callable[[Action], float]] = None) -> RepairResult:
        original = Action.from_obj(action)
        first = self.gate.evaluate(original, blueprint)
        if first.allowed:
            return RepairResult(original, False, 0.0, [])
        utility = utility or (lambda candidate: 1.0 if candidate.tool == original.tool else 0.5)
        candidates: List[Tuple[float, Action]] = []
        recipients = list(getattr(blueprint, "allowed_recipients", []))
        candidate_tools: List[str] = []
        if original.tool in blueprint.authorized_tools:
            candidate_tools.append(original.tool)
        candidate_tools.extend(
            tool for tool in self.tool_substitutions.get(original.tool, ())
            if tool in blueprint.authorized_tools and tool not in candidate_tools
        )
        for tool in candidate_tools:
            args = dict(original.arguments)
            changes = 0
            destination_key = next((k for k in ("to", "recipient", "destination") if k in args), None)
            if destination_key and recipients and str(args[destination_key]) not in recipients:
                args[destination_key] = recipients[0]
                changes += 1
            spec = blueprint.allowed_tools[tool]
            retained = {k: v for k, v in args.items() if k in spec.parameter_rules}
            changes += len(args) - len(retained)
            args = retained
            for name, rule in spec.parameter_rules.items():
                if name in args and rule.allowed_values and args[name] not in rule.allowed_values:
                    args[name] = rule.allowed_values[0]
                    changes += 1
                elif name not in args and rule.required and rule.allowed_values:
                    args[name] = rule.allowed_values[0]
                    changes += 1
            candidate = Action(tool, args)
            decision = self.gate.evaluate(candidate, blueprint)
            if decision.allowed and utility(candidate) >= self.utility_threshold:
                cost = (0 if tool == original.tool else self.tool_cost) + changes * self.argument_cost
                candidates.append((cost, candidate))
        if not candidates:
            return RepairResult(None, False, math.inf, first.violations, abstained=True)
        cost, best = min(candidates, key=lambda item: (item[0], item[1].tool))
        return RepairResult(best, True, cost, first.violations)


@dataclass
class ConstrainedGenerationResult:
    action: Optional[Action]
    attempts: int
    first_pass_valid: bool
    repaired: bool
    abstained: bool
    violations: List[str] = field(default_factory=list)
    raw_outputs: List[Any] = field(default_factory=list)


class BlueprintConstrainedActionGenerator:
    """Backend-neutral implementation of blueprint-constrained generation.

    A local CFG backend (for example XGrammar/llguidance/Outlines) or an API
    strict-structured-output adapter receives the compiled schema. Every parsed
    action is still checked by the controller before it can reach a tool.
    """

    def __init__(
        self,
        tracker: StatefulAuthorityTracker,
        backend: Callable[..., Any],
        repairer: Optional[MinimumChangeRepairer] = None,
        schema_compiler: Optional[BlueprintGrammarCompiler] = None,
    ):
        self.tracker = tracker
        self.backend = backend
        self.repairer = repairer
        self.schema_compiler = schema_compiler or BlueprintGrammarCompiler()

    def generate(
        self,
        prompt: str,
        envelope: ProvenanceEnvelope,
        *,
        max_attempts: int = 1,
        allow_repair: bool = True,
    ) -> ConstrainedGenerationResult:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        schema = self.schema_compiler.compile_json_schema(self.tracker.current)
        raw_outputs: List[Any] = []
        last_violations: List[str] = []
        for attempt in range(1, max_attempts + 1):
            raw = self.backend(prompt=prompt, json_schema=schema, attempt=attempt)
            raw_outputs.append(raw)
            try:
                action = Action.from_obj(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_violations = [f"action:invalid_structure:{exc}"]
                continue
            decision = self.tracker.inspect_action(action, envelope)
            if decision.allowed:
                return ConstrainedGenerationResult(
                    action, attempt, attempt == 1, False, False, [], raw_outputs
                )
            last_violations = decision.violations
            if allow_repair and self.repairer is not None:
                repair = self.repairer.repair(action, self.tracker.current)
                if repair.action is not None:
                    repaired_decision = self.tracker.inspect_action(repair.action, envelope)
                    if repaired_decision.allowed:
                        return ConstrainedGenerationResult(
                            repair.action, attempt, False, True, False,
                            decision.violations, raw_outputs,
                        )
                    last_violations = repaired_decision.violations
        return ConstrainedGenerationResult(
            None, max_attempts, False, False, True, last_violations, raw_outputs
        )


class SafeToolExecutor:
    """The only supported bridge from model output to a side-effecting tool.

    Validation and optional repair happen before ``dispatch`` is called.  This
    ordering is the executable form of the pre-execution safety invariant.
    """

    def __init__(self, tracker: StatefulAuthorityTracker,
                 dispatch: Callable[[str, Dict[str, Any]], Any],
                 repairer: Optional[MinimumChangeRepairer] = None):
        self.tracker = tracker
        self.dispatch = dispatch
        self.repairer = repairer

    def execute(self, action: Any, provenance: ProvenanceEnvelope,
                allow_repair: bool = True) -> Dict[str, Any]:
        parsed = Action.from_obj(action)
        decision = self.tracker.inspect_action(parsed, provenance)
        repair: Optional[RepairResult] = None
        if not decision.allowed and allow_repair and self.repairer is not None:
            repair = self.repairer.repair(parsed, self.tracker.current)
            if repair.action is not None:
                parsed = repair.action
                decision = self.tracker.inspect_action(parsed, provenance)
        if not decision.allowed:
            return {"executed": False, "verdict": "BLOCK", "violations": decision.violations,
                    "repair": repair}
        result = self.dispatch(parsed.tool, dict(parsed.arguments))
        self.tracker.record_dispatch(parsed)
        return {"executed": True, "verdict": "PASS", "action": parsed.to_dict(),
                "result": result, "repair": repair}


class GuardedToolRuntime:
    """Drop-in guard for runtimes exposing ``run_function`` and ``functions``.

    It makes the reference monitor unavoidable at the side-effect boundary,
    including nested tool calls embedded in arguments.
    """

    def __init__(
        self,
        runtime: Any,
        tracker: StatefulAuthorityTracker,
        *,
        authenticated_user_confirmation: bool = False,
    ):
        self.runtime = runtime
        self.tracker = tracker
        self.authenticated_user_confirmation = authenticated_user_confirmation
        self.blocked_calls: List[Dict[str, Any]] = []

    @property
    def functions(self) -> Any:
        # Capability hiding is defense in depth for API tool calling: the model
        # only receives schemas for currently authorized tools. The execution
        # check below remains mandatory even if a caller forges a tool call.
        allowed = set(self.tracker.current.authorized_tools)
        return {
            name: function for name, function in self.runtime.functions.items()
            if name in allowed
        }

    def register_function(self, function: Any) -> Any:
        return self.runtime.register_function(function)

    def _walk_actions(self, tool: str, arguments: Mapping[str, Any]) -> List[Action]:
        actions: List[Action] = []
        for value in arguments.values():
            nested_tool = getattr(value, "function", None)
            nested_args = getattr(value, "args", None)
            if nested_tool is not None and isinstance(nested_args, Mapping):
                actions.extend(self._walk_actions(str(nested_tool), nested_args))
        actions.append(Action(tool, dict(arguments)))
        return actions

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        actions = self._walk_actions(function, kwargs)
        envelope = ProvenanceEnvelope(
            actions,
            Provenance.UNTRUSTED_TOOL,
            source="model_tool_call",
            authenticated=self.authenticated_user_confirmation,
            tainted_context=any(item.get("type") == "observation" for item in self.tracker.history),
        )
        for action in actions:
            decision = self.tracker.inspect_action(action, envelope)
            if not decision.allowed:
                event = {"action": action.to_dict(), "violations": decision.violations}
                self.blocked_calls.append(event)
                message = "ToolFence_BLOCKED: " + "; ".join(decision.violations)
                if raise_on_error:
                    raise PermissionError(message)
                return "", message

        result = self.runtime.run_function(env, function, kwargs, raise_on_error=raise_on_error)
        value, error = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        if error is None:
            for action in actions:
                self.tracker.record_dispatch(action)
            self.tracker.observe(value, Provenance.UNTRUSTED_TOOL, source=function)
        return result
