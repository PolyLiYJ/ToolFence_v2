"""Deterministic capability and data-flow enforcement."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from toolfence.core.monitor import CoreDecision, CoreLease, CoreMonitor
from toolfence.foundation.authorization import Action, ProvenanceEnvelope
from toolfence.foundation.blueprint import ParameterRule, ToolPermission

from .schema import AuthorizationPlan, Capability, ParameterBinding, ToolManifest


READ_EFFECTS = {"read", "local_compute"}


def _is_temporal_parameter(name: str, parameter: Any) -> bool:
    role = getattr(parameter, "semantic_role", "") if parameter is not None else ""
    tokens = set(__import__("re").findall(r"[a-z0-9]+", f"{name} {role}".casefold()))
    return bool(tokens & {
        "date", "day", "time", "datetime", "timestamp", "temporal",
        "deadline", "startdate", "enddate", "starttime", "endtime",
    })


def _scalars(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _scalars(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _scalars(item)
        return
    if hasattr(value, "model_dump"):
        yield from _scalars(value.model_dump())
        return
    if hasattr(value, "dict"):
        yield from _scalars(value.dict())
        return
    yield str(value)


def _field_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            for scalar in _scalars(item):
                yield path, scalar
            yield from _field_scalars(item, path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _field_scalars(item, prefix)


def _canonical_literal(value: Any) -> Any:
    if isinstance(value, str):
        result = value.strip()
        while len(result) >= 2 and result[0] == result[-1] and result[0] in {"'", '"'}:
            result = result[1:-1].strip()
        return result
    if isinstance(value, list):
        return [_canonical_literal(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonical_literal(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_literal(item) for key, item in value.items()}
    return value


def _literal_equivalent(expected: Any, actual: Any) -> bool:
    """Compare controller literals while tolerating schema-only singleton arrays."""
    expected = _canonical_literal(expected)
    actual = _canonical_literal(actual)
    if expected == actual:
        return True
    if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, list):
        return expected[0] == actual
    if isinstance(actual, list) and len(actual) == 1 and not isinstance(expected, list):
        return actual[0] == expected
    return False


def _template_matches(template: Any, value: Any) -> bool:
    if not isinstance(template, str) or not isinstance(value, str):
        return False
    parts = []
    cursor = 0
    for match in __import__("re").finditer(r"\{[^{}]+\}", template):
        parts.append(__import__("re").escape(template[cursor:match.start()]))
        parts.append(r".+?")
        cursor = match.end()
    parts.append(__import__("re").escape(template[cursor:]))
    pattern = "^" + "".join(parts) + "$"
    return __import__("re").match(pattern, value, flags=__import__("re").IGNORECASE) is not None


def _query_supports_value(query: str, value: Any) -> bool:
    normalized = " ".join(query.casefold().split())
    scalars = [" ".join(item.casefold().split()) for item in _scalars(value)]
    return bool(scalars) and all(item and item in normalized for item in scalars)


def _derived_scalar_matches(value: str, candidate: str) -> bool:
    value = " ".join(value.casefold().split())
    candidate = " ".join(candidate.casefold().split())
    if not value or not candidate:
        return False
    return value == candidate or value in candidate


class SemanticMonitor:
    """Enforce the model-compiled plan without consulting an LLM at runtime."""

    def __init__(self, blueprint, plan: AuthorizationPlan, manifest: ToolManifest):
        self.core = CoreMonitor(blueprint)
        self.plan = plan
        self.manifest = manifest
        self.manifest_by_name = manifest.by_name()
        self.authenticated_query = ""
        # Keep distinct authority templates for the same tool. Never merge
        # parameter bindings across templates, which could create new authority.
        self.capabilities = plan.by_tool_all()
        self.observed_values: Dict[str, Set[str]] = defaultdict(set)
        self.observed_fields: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.executed_tools: Set[str] = set()
        self.audit_log: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self._proposal_step = 0
        self._lease_events: Dict[str, dict[str, Any]] = {}

        # Per-capability/template budgets. CoreMonitor still enforces the
        # aggregate per-tool/session ceiling from the blueprint, while these
        # counters prevent one legal template from consuming another template's
        # authority (e.g. London twice instead of London once + Paris once).
        self._capability_keys: Dict[int, str] = {}
        self._capability_limits: Dict[str, int] = {}
        self._capability_committed: Dict[str, int] = defaultdict(int)
        self._capability_pending: Dict[str, int] = defaultdict(int)
        self._lease_capabilities: Dict[str, str] = {}
        for tool, templates in self.capabilities.items():
            for index, capability in enumerate(templates):
                key = f"{tool}#{index}"
                self._capability_keys[id(capability)] = key
                self._capability_limits[key] = int(capability.max_calls)

    @property
    def blueprint(self):
        return self.core.blueprint

    def register_user_query(self, query: str) -> None:
        self.authenticated_query = query
        self.core.register_user_query(query)

    def grant_capability(
        self,
        capability: Capability,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Extend the running blueprint with a judge-authorized capability.

        The granted capability is appended to ``self.capabilities[tool]`` and
        registered with its own budget key, so subsequent calls of the same
        shape are matched by ``_match_capability`` on the fast path -- with no
        further judge call. This is the "dynamic blueprint": the plan grows at
        runtime, one judge-authorized capability at a time, instead of judging
        every denied call.
        """
        tool = capability.tool
        templates = self.capabilities.setdefault(tool, [])
        # Avoid duplicate shapes (same parameter binding kinds + sources).
        signature = self._capability_signature(capability)
        for existing in templates:
            if self._capability_signature(existing) == signature:
                return  # already granted
        templates.append(capability)
        key = f"{tool}#{len(templates) - 1}"
        self._capability_keys[id(capability)] = key
        self._capability_limits[key] = int(capability.max_calls)

        # Keep the lower-level CoreMonitor in sync with the dynamic semantic
        # grant. Previously only the semantic capability graph was extended;
        # CoreMonitor still saw the tool as DENIED (or retained too small a
        # budget), so an approved grant immediately failed on re-prepare.
        spec = self.blueprint.allowed_tools.get(tool)
        if spec is not None:
            was_allowed = spec.permission == ToolPermission.ALLOWED
            spec.permission = ToolPermission.ALLOWED
            for name, value in (arguments or {}).items():
                if name in spec.parameter_rules:
                    continue
                if isinstance(value, bool):
                    type_name = "boolean"
                elif isinstance(value, int):
                    type_name = "integer"
                elif isinstance(value, float):
                    type_name = "number"
                elif isinstance(value, list):
                    type_name = "array"
                elif isinstance(value, dict):
                    type_name = "object"
                else:
                    type_name = "string"
                spec.parameter_rules[name] = ParameterRule(
                    name=name, type=type_name, required=True,
                    secret_policy="allow",
                )
            increment = max(1, int(capability.max_calls))
            spec.max_calls_per_session = (
                int(spec.max_calls_per_session) + increment
                if was_allowed else increment
            )
            self.blueprint.max_total_tool_calls = (
                int(self.blueprint.max_total_tool_calls) + increment
            )
        self.audit_log.append({
            "event": "capability_grant",
            "tool": tool,
            "effect": capability.effect,
            "origin": capability.origin,
            "reusable": capability.reusable,
            "max_calls": capability.max_calls,
            "parameters": {
                name: {
                    "binding": binding.binding,
                    "source_tools": list(binding.source_tools),
                    "allowed_fields": list(binding.allowed_fields),
                }
                for name, binding in capability.parameters.items()
            },
        })

    @staticmethod
    def _capability_signature(capability: Capability) -> tuple:
        parts = []
        for name in sorted(capability.parameters):
            binding = capability.parameters[name]
            parts.append((
                name, binding.binding,
                json.dumps(binding.value, sort_keys=True, default=str)
                if binding.binding in {"literal", "normalized_literal", "template"}
                else None,
                tuple(sorted(binding.source_tools)),
                tuple(sorted(binding.allowed_fields)),
            ))
        return (capability.tool, capability.effect, tuple(parts))

    @staticmethod
    def _source_label(binding: Optional[ParameterBinding]) -> str:
        if binding is None:
            return "undeclared"
        return {
            "literal": "authenticated_user_literal",
            "normalized_literal": "authenticated_user_normalized_literal",
            "template": "authenticated_user_template",
            "derived": "authorized_tool_output",
            "computed": "authorized_tool_computation",
            "generated": "model_generated_non_authority",
            "free": "agent_selected_non_authority",
        }[binding.binding]

    def _parameter_sources(
        self, tool: str, arguments: Mapping[str, Any], capability=None,
    ) -> Dict[str, Any]:
        if capability is None:
            candidates = self.capabilities.get(tool, [])
            capability = candidates[0] if candidates else None
        bindings = capability.parameters if capability is not None else {}
        result: Dict[str, Any] = {}
        for name in arguments:
            binding = bindings.get(name)
            result[name] = {
                "label": self._source_label(binding),
                "binding": binding.binding if binding is not None else None,
                "source_tools": list(binding.source_tools) if binding is not None else [],
                "allowed_fields": list(binding.allowed_fields) if binding is not None else [],
            }
        return result

    def _capability_violations(
        self, capability, tool: str, arguments: Mapping[str, Any],
    ) -> list[str]:
        violations: list[str] = []
        for name, value in arguments.items():
            binding = capability.parameters.get(name)
            if binding is None:
                violations.append(f"binding:{tool}.{name}:not_declared")
                continue
            violation = self._binding_violation(tool, name, value, binding)
            if violation:
                violations.append(violation)
        return violations

    def _matching_sources(self, value: Any, allowed_sources: Iterable[str]) -> list[str]:
        matched = []
        values = list(_scalars(value))
        for source in allowed_sources:
            if source not in self.executed_tools:
                continue
            candidates = self.observed_values.get(source, set())
            if values and all(
                any(_derived_scalar_matches(item, candidate) for candidate in candidates)
                for item in values
            ):
                matched.append(source)
        return matched

    def _dynamic_read_refinement(
        self, tool: str, arguments: Mapping[str, Any], candidates: list[Capability],
    ) -> Optional[Capability]:
        """Refine an already-authorized read edge without creating new authority.

        Dynamic refinement is deliberately limited to a tool that already has at
        least one static/synthesized read capability.  For authority-sensitive
        parameters, the runtime value must either be grounded in the authenticated
        user request or follow a source edge that the static plan already allowed.
        Non-sensitive read parameters may vary freely.  Side effects never enter
        this path.
        """
        semantic = self.manifest_by_name.get(tool)
        if semantic is None or semantic.effect not in READ_EFFECTS or not candidates:
            return None
        bindings: Dict[str, ParameterBinding] = {}
        for name, value in arguments.items():
            parameter = semantic.parameters.get(name)
            if parameter is None:
                return None
            planned = [cap.parameters.get(name) for cap in candidates if name in cap.parameters]
            planned = [binding for binding in planned if binding is not None]

            if _query_supports_value(self.authenticated_query, value):
                bindings[name] = ParameterBinding(binding="literal", value=value)
                continue

            # Reuse only provenance edges that were already present in the plan.
            allowed_sources = []
            allowed_fields = []
            for binding in planned:
                if binding.binding in {"derived", "computed", "template"}:
                    allowed_sources.extend(binding.source_tools)
                    allowed_fields.extend(binding.allowed_fields)
            matched_sources = self._matching_sources(value, dict.fromkeys(allowed_sources))
            if matched_sources:
                bindings[name] = ParameterBinding(
                    binding="derived",
                    source_tools=matched_sources,
                    allowed_fields=list(dict.fromkeys(allowed_fields)),
                )
                continue

            # A read filter is still capable of destroying utility when the
            # model invents a missing year or time. Only an authenticated literal
            # or declared source edge may refine temporal parameters dynamically.
            if _is_temporal_parameter(name, parameter):
                return None

            if not parameter.authority_sensitive:
                bindings[name] = ParameterBinding(binding="free")
                continue
            return None

        return Capability(
            tool=tool,
            effect=semantic.effect,
            operation=semantic.operation,
            reusable=True,
            origin="runtime_refinement",
            max_calls=1,
            parameters=bindings,
        )

    def _match_capability(
        self, tool: str, arguments: Mapping[str, Any],
    ):
        """Return a matching authority template or a safe read refinement."""
        candidates = self.capabilities.get(tool, [])
        if not candidates:
            return None, [f"capability:{tool}:not_authorized"]
        best = None
        best_violations = None
        exhausted_match = None
        for capability in candidates:
            violations = self._capability_violations(capability, tool, arguments)
            if not violations:
                # Read/local-compute authority is reusable.  Resource exhaustion
                # remains enforced by CoreMonitor's per-tool/session ceilings.
                if capability.reusable or capability.effect in READ_EFFECTS:
                    return capability, []
                key = self._capability_keys[id(capability)]
                used = self._capability_committed[key] + self._capability_pending[key]
                if used < self._capability_limits[key]:
                    return capability, []
                exhausted_match = capability
                continue
            if best_violations is None or len(violations) < len(best_violations):
                best = capability
                best_violations = violations

        refined = self._dynamic_read_refinement(tool, arguments, candidates)
        if refined is not None:
            return refined, []
        if exhausted_match is not None:
            key = self._capability_keys[id(exhausted_match)]
            return exhausted_match, [f"budget:capability:{key}:exhausted"]
        return best, list(dict.fromkeys(best_violations or []))

    def _binding_violation(
        self, tool: str, name: str, value: Any, binding: ParameterBinding,
    ) -> Optional[str]:
        if binding.binding in {"literal", "normalized_literal"}:
            if not _literal_equivalent(binding.value, value):
                suffix = (
                    "normalized_literal_mismatch"
                    if binding.binding == "normalized_literal"
                    else "literal_mismatch"
                )
                return f"binding:{tool}.{name}:{suffix}"
            return None
        if binding.binding == "template":
            if not _template_matches(binding.value, value):
                return f"binding:{tool}.{name}:template_mismatch"
            missing_sources = set(binding.source_tools) - self.executed_tools
            if missing_sources:
                return f"binding:{tool}.{name}:source_not_executed"
            return None
        missing_sources = set(binding.source_tools) - self.executed_tools
        if missing_sources:
            return f"binding:{tool}.{name}:source_not_executed"
        if binding.binding == "derived":
            if binding.allowed_fields:
                candidates: Set[str] = set()
                for source in binding.source_tools:
                    fields = self.observed_fields.get(source, {})
                    for allowed in binding.allowed_fields:
                        for path, field_values in fields.items():
                            if path == allowed:
                                candidates.update(field_values)
            else:
                candidates = set().union(*(
                    self.observed_values.get(source, set()) for source in binding.source_tools
                ))
            # Only an unconstrained source may use unstructured values.
            # A declared field path must never fall back to another field.
            if not candidates and not binding.allowed_fields:
                candidates = set().union(*(
                    self.observed_values.get(source, set()) for source in binding.source_tools
                ))
            values = set(_scalars(value))
            if not values or not all(
                any(_derived_scalar_matches(item, candidate) for candidate in candidates)
                for item in values
            ):
                return f"binding:{tool}.{name}:not_derived_from_authorized_source"
        return None

    def contraction_candidate(
        self, action: Action, violations: Iterable[str],
    ) -> Optional[Action]:
        """Drop only undeclared arguments; never invent or substitute values."""
        candidates = self.capabilities.get(action.tool, [])
        violation_list = list(violations)
        if not candidates or not violation_list:
            return None
        prefix = f"binding:{action.tool}."
        if any(not (code.startswith(prefix) and code.endswith(":not_declared"))
               for code in violation_list):
            return None
        # Try contraction against each call template independently; return only a
        # contraction that already satisfies one complete template.
        for capability in candidates:
            retained = {
                name: value for name, value in action.arguments.items()
                if name in capability.parameters
            }
            if retained == dict(action.arguments):
                continue
            if not self._capability_violations(capability, action.tool, retained):
                return Action(action.tool, retained)
        return None

    def consolidate_successful_repair(
        self,
        *,
        original_event_index: int,
        repaired: Action,
        repaired_decision: CoreDecision,
    ) -> None:
        """Represent a contraction retry as one auditable proposed action."""
        if repaired_decision.lease is None:
            raise ValueError("Cannot consolidate a repair without an authorization lease")
        original = self.action_trace[original_event_index]
        retry = self.action_trace.pop()
        removed = sorted(
            set(original["proposed_action"]["arguments"]) - set(repaired.arguments)
        )
        original.update({
            "effective_action": repaired.to_dict(),
            "effective_parameter_sources": self._parameter_sources(
                repaired.tool, repaired.arguments,
                capability=self._match_capability(repaired.tool, repaired.arguments)[0],
            ),
            "decision": "repair",
            "allowed": True,
            "repair": {
                "attempted": True,
                "applied": True,
                "kind": "contraction_only",
                "removed_parameters": removed,
                "initial_violation_codes": list(original["violation_codes"]),
            },
            "execution_status": "repaired_pending_dispatch",
        })
        self._lease_events[repaired_decision.lease.lease_id] = original
        # Preserve the internal retry only in the legacy diagnostic stream;
        # action_trace remains one record per model-proposed action.
        self.audit_log.append({
            "event": "repair",
            "action": original["proposed_action"],
            "effective_action": repaired.to_dict(),
            "removed_parameters": removed,
            "retry_allowed": retry["allowed"],
        })

    def prepare(
        self, action: Any, envelope: ProvenanceEnvelope, *, lease_ttl_seconds: float = 30.0,
    ) -> CoreDecision:
        parsed = Action.from_obj(action)
        self._proposal_step += 1
        capability, violations = self._match_capability(parsed.tool, parsed.arguments)
        if violations:
            decision = CoreDecision(False, tuple(dict.fromkeys(violations)))
        else:
            decision = self.core.prepare(
                parsed, envelope, lease_ttl_seconds=lease_ttl_seconds,
            )
        provenance = {
            "label": envelope.provenance.value,
            "source": envelope.source,
            "authenticated": envelope.authenticated,
            "tainted_context": envelope.tainted_context,
        }
        trace_event = {
            "step": self._proposal_step,
            "event": "authorization",
            "proposed_action": parsed.to_dict(),
            "effective_action": parsed.to_dict() if decision.allowed else None,
            "decision": "allow" if decision.allowed else "deny",
            "allowed": decision.allowed,
            "violation_codes": list(decision.violations),
            "parameter_sources": self._parameter_sources(
                parsed.tool, parsed.arguments, capability=capability,
            ),
            "proposal_provenance": provenance,
            "authority_refinement": (
                {
                    "kind": "runtime_read_refinement",
                    "tool": parsed.tool,
                    "non_expanding": True,
                }
                if capability is not None and getattr(capability, "origin", None) == "runtime_refinement"
                else None
            ),
            "repair": {
                "attempted": False,
                "applied": False,
                "reason": "toolfence_semantic_has_no_action_repair",
            },
            "execution_status": (
                "authorized_pending_dispatch" if decision.allowed else "blocked_before_dispatch"
            ),
        }
        self.action_trace.append(trace_event)
        if decision.lease is not None:
            self._lease_events[decision.lease.lease_id] = trace_event
            if (
                capability is not None
                and not capability.reusable
                and id(capability) in self._capability_keys
            ):
                key = self._capability_keys[id(capability)]
                self._capability_pending[key] += 1
                self._lease_capabilities[decision.lease.lease_id] = key
                trace_event["capability_budget"] = {
                    "key": key,
                    "limit": self._capability_limits[key],
                    "committed_before": self._capability_committed[key],
                    "pending_after": self._capability_pending[key],
                }
        if capability is not None and getattr(capability, "origin", None) == "runtime_refinement":
            self.audit_log.append({
                "event": "authority_refinement",
                "kind": "runtime_read_refinement",
                "tool": parsed.tool,
                "action": parsed.to_dict(),
                "non_expanding": True,
            })
        # Keep the legacy event keys for readers of earlier result files.
        self.audit_log.append({
            "event": "authorization",
            "action": parsed.to_dict(),
            "allowed": decision.allowed,
            "violations": list(decision.violations),
            "step": self._proposal_step,
        })
        return decision

    def commit(self, lease: CoreLease) -> None:
        self.core.commit(lease)
        self.executed_tools.add(lease.action.tool)
        key = self._lease_capabilities.pop(lease.lease_id, None)
        if key is not None:
            self._capability_pending[key] = max(0, self._capability_pending[key] - 1)
            self._capability_committed[key] += 1
        event = self._lease_events.pop(lease.lease_id, None)
        if event is not None:
            event["execution_status"] = "committed"
            if key is not None:
                event.setdefault("capability_budget", {}).update({
                    "committed_after": self._capability_committed[key],
                    "pending_after_commit": self._capability_pending[key],
                })

    def rollback(self, lease: CoreLease, *, reason: str = "runtime_rollback") -> None:
        self.core.rollback(lease)
        key = self._lease_capabilities.pop(lease.lease_id, None)
        if key is not None:
            self._capability_pending[key] = max(0, self._capability_pending[key] - 1)
        event = self._lease_events.pop(lease.lease_id, None)
        if event is not None:
            event["execution_status"] = "rolled_back"
            event["rollback_reason"] = reason
            if key is not None:
                event.setdefault("capability_budget", {}).update({
                    "committed_after": self._capability_committed[key],
                    "pending_after_rollback": self._capability_pending[key],
                })

    def observe(self, value: Any, source: str) -> None:
        self.core.observe(value, source)
        self.executed_tools.add(source)
        self.observed_values[source].update(_scalars(value))
        for field, scalar in _field_scalars(value):
            self.observed_fields[source][field].add(scalar)
        self.audit_log.append({
            "event": "observation",
            "source": source,
            "value_sha256": __import__("hashlib").sha256(
                json.dumps(value, sort_keys=True, default=str).encode()
            ).hexdigest(),
        })

    def metrics(self) -> Dict[str, Any]:
        metrics = self.core.metrics()
        metrics["semantic_audit_events"] = len(self.audit_log)
        return metrics
