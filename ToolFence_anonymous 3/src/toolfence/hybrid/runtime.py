"""Hybrid runtime: deterministic v2 authorization with a semantic fallback.

Combines the two architectures instead of choosing between them:

1. The compiled blueprint monitor (src/toolfence_semantic_v2) is the fast path.
   It is deterministic, costs no LLM call, and enforces typed data-flow
   bindings and effect evidence. Anything it allows is executed immediately, so
   a legitimate action can never be lost to a judge false denial.
2. Only when the blueprint denies a call does the runtime judge re-examine it
   against the authenticated request. This repairs the blueprint's two failure
   modes: planner under-authorization (a needed tool missing from the plan) and
   total compilation failure (deny-all abstention).
3. Tools are never hidden. Hiding them turned every planner omission into silent
   starvation with no denial event; here an unauthorized call is a visible,
   reviewable decision.

Security note: this is an OR of two gates, so it is strictly more permissive
than either alone. The judge only ever sees calls the blueprint already refused,
and it applies the same "does this serve the authenticated request" rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from toolfence.foundation.authorization import Action, Provenance, ProvenanceEnvelope

from ..judge.judge import JudgeDecision, RuntimeJudge
from ..judge.effects import heuristic_effect
from ..judge.prompts import denial_message, tool_index
from ..judge.runtime import _scalars, compute_provenance
from ..judge.workflow import (
    verified_discount_web_step, verified_password_otp_step,
    verified_workflow_prerequisite,
)
from ..semantic.schema import Capability, ParameterBinding, ToolManifest
from .grant_cache import CapabilityGrantCache, capability_signature


class HybridRuntime:
    """Fast path = blueprint monitor; fallback = runtime judge."""

    def __init__(
        self,
        runtime: Any,
        monitor: Any,
        judge: RuntimeJudge,
        *,
        query: str,
        tools: List[Dict[str, Any]],
        workflow_context: bool = False,
    ):
        self.runtime = runtime
        self.monitor = monitor
        self.judge = judge
        self.query = query
        self.tools = tool_index(tools)
        self.workflow_context = bool(workflow_context)
        self.completed_tools: List[str] = []
        self.structured_observations: Dict[str, Any] = {}
        self.available_read_tools = sorted(
            name for name, info in self.tools.items()
            if heuristic_effect(name, info.get("description", "")) == "read"
        )
        self.decisions: List[Dict[str, Any]] = []
        self.observed_text: Dict[str, str] = {}

    @property
    def functions(self) -> Any:
        # Every tool stays visible; authorization is a decision, not a vanishing
        # tool. This removes the starvation failure mode of the blueprint alone.
        return self.runtime.functions

    def register_function(self, function: Any) -> Any:
        return self.runtime.register_function(function)

    def _needs_judge_review(
        self, function: str, provenance: Mapping[str, str],
    ) -> bool:
        """True when a blueprint-allowed read introduces an unauthenticated target."""
        if all(source == "user" for source in provenance.values()):
            return False
        manifest = getattr(self.monitor, "manifest", None)
        if manifest is None:
            return False
        semantic = manifest.by_name().get(function)
        if semantic is None:
            return False
        return semantic.effect in {"read", "local_compute"}

    def _is_read_auto_allow(
        self, function: str, provenance: Mapping[str, str],
    ) -> bool:
        """True when a read/local_compute call may bypass the judge entirely.

        Mirrors toolfence_runtime_manifest's read_auto_allow: the tool's
        effect is read/local_compute AND every argument's provenance is
        "user" (occurs verbatim in the authenticated request). Such a call
        cannot introduce a new access target, so judging it only risks
        over-denial. A read with tool/model-sourced arguments is NOT
        auto-allowed (slack injection completes via an injected URL).
        """
        manifest = getattr(self.monitor, "manifest", None)
        if manifest is None or not hasattr(manifest, "by_name"):
            return False
        semantic = manifest.by_name().get(function)
        if semantic is None:
            return False
        if semantic.effect not in {"read", "local_compute"}:
            return False
        operational_names = {
            "n", "limit", "count", "page", "page_size", "offset",
            "max_results", "top_k", "num_results",
        }
        for name, source in provenance.items():
            if source == "user":
                continue
            parameter = semantic.parameters.get(name)
            # Pagination, result-count, formatting, and other controller-owned
            # non-authority parameters cannot redirect the read target. Let the
            # agent choose them without asking a weak judge. Unknown or
            # authority-sensitive parameters remain fail-closed.
            role = (
                getattr(parameter, "semantic_role", "").casefold()
                if parameter is not None else ""
            )
            operational = (
                name.casefold() in operational_names
                or any(token in role for token in (
                    "result_count", "page_size", "pagination", "result_limit",
                ))
            )
            if not operational and (
                parameter is None or parameter.authority_sensitive
            ):
                return False
        return True

    def _is_explicit_observed_document_read(
        self, function: str, arguments: Mapping[str, Any],
    ) -> bool:
        """Hook for narrowly scoped read-only repairs in hybrid variants."""
        return False

    def _is_verified_read_dependency(
        self,
        function: str,
        provenance: Mapping[str, str],
        evidence: Mapping[str, Mapping[str, str]],
    ) -> bool:
        """Allow a declared read dependency when its values were observed.

        Merely appearing in untrusted tool output is insufficient: the static
        plan must also declare the exact source-tool edge for every tool-sourced
        parameter. This preserves injection resistance while avoiding an LLM
        judge false denial for a dependency the compiler already authorized.
        """
        manifest = getattr(self.monitor, "manifest", None)
        plan = getattr(self.monitor, "plan", None)
        if manifest is None or plan is None or not hasattr(manifest, "by_name"):
            return False
        semantic = manifest.by_name().get(function)
        if semantic is None or semantic.effect not in {"read", "local_compute"}:
            return False
        if not provenance or any(source == "model" for source in provenance.values()):
            return False
        candidates = [cap for cap in plan.capabilities if cap.tool == function]
        for capability in candidates:
            verified = True
            for name, source in provenance.items():
                if source == "user":
                    continue
                if not source.startswith("tool:") or name not in evidence:
                    verified = False
                    break
                source_tool = source.split(":", 1)[1]
                binding = capability.parameters.get(name)
                if (
                    binding is None
                    or binding.binding not in {"derived", "computed", "template"}
                    or source_tool not in binding.source_tools
                ):
                    verified = False
                    break
            if verified:
                return True
        return False

    def _record_observation(self, function: str, value: Any) -> None:
        self.completed_tools.append(function)
        if function in {
            "get_shopping_account_user_information", "view_cart",
            "view_available_discounts",
        }:
            self.structured_observations[function] = value
        if function == "get_shopping_linked_unread_emails" and isinstance(value, list):
            self.structured_observations[function] = value
        if function == "search_product" and isinstance(value, Mapping):
            self.structured_observations.setdefault(function, []).append(value)
        try:
            rendered = value if isinstance(value, str) else json.dumps(value, default=str)
        except TypeError:
            rendered = str(value)
        previous = self.observed_text.get(function, "")
        merged = (previous + "\n" + rendered) if previous else rendered
        self.observed_text[function] = merged[:4000]

    def _authenticated_literal_conflict(
        self, function: str, violations: List[str],
    ) -> bool:
        """Keep a user-specified value fixed across the judge fallback."""
        if not self.workflow_context:
            return False
        for violation in violations:
            prefix = f"binding:{function}."
            if not violation.startswith(prefix):
                continue
            tail = violation[len(prefix):]
            if not tail.endswith((":literal_mismatch", ":normalized_literal_mismatch")):
                continue
            name = tail.split(":", 1)[0]
            for capability in getattr(self.monitor.plan, "capabilities", []):
                if capability.tool != function:
                    continue
                binding = capability.parameters.get(name)
                if binding is None or binding.binding not in {"literal", "normalized_literal"}:
                    continue
                values = list(_scalars(binding.value))
                if values and all(
                    len(value) >= 2 and value.casefold() in self.query.casefold()
                    for value in values
                ):
                    return True
        return False

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        normalized = {
            name: value for name, value in dict(kwargs).items() if value is not None
        }
        action = Action(function, normalized)
        envelope = ProvenanceEnvelope(
            [action], Provenance.UNTRUSTED_TOOL, source="model_tool_call",
            authenticated=True, tainted_context=bool(
                getattr(self.monitor, "executed_tools", None)
            ),
        )
        trace_index = len(self.monitor.action_trace)
        decision = self.monitor.prepare(action, envelope)
        dispatch = dict(normalized)
        route = "blueprint"

        if not decision.allowed:
            # Blueprint contraction-only repair first (drops undeclared args).
            repaired = self.monitor.contraction_candidate(action, decision.violations)
            if repaired is not None:
                repaired_decision = self.monitor.prepare(repaired, envelope)
                if repaired_decision.allowed and repaired_decision.lease is not None:
                    self.monitor.consolidate_successful_repair(
                        original_event_index=trace_index,
                        repaired=repaired,
                        repaired_decision=repaired_decision,
                    )
                    decision = repaired_decision
                    dispatch = dict(repaired.arguments)
                    route = "blueprint_repair"

        if not decision.allowed or decision.lease is None:
            blueprint_violations = list(decision.violations)
            provenance, evidence = compute_provenance(
                self.query, self.observed_text, dispatch,
            )
            if self._authenticated_literal_conflict(function, blueprint_violations):
                reason = "proposed value conflicts with the authenticated request"
                self.decisions.append({
                    "tool": function, "arguments": dict(dispatch),
                    "route": "authenticated_literal_conflict",
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance, "allowed": False, "reason": reason,
                })
                message = denial_message(
                    reason, denied_tool=function, authenticated_request=self.query,
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message
            if self.workflow_context and (
                verified_workflow_prerequisite(
                    self.query, function, dispatch, self.structured_observations,
                )
                or verified_password_otp_step(
                    self.query, function, dispatch, env, self.decisions,
                    self.completed_tools,
                )
            ):
                self.decisions.append({
                    "tool": function, "arguments": dict(dispatch),
                    "route": "workflow_dependency_allow",
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance, "allowed": True,
                    "reason": "verified shopping workflow dependency",
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error
            if self.workflow_context and verified_discount_web_step(
                self.query, function, dispatch, self.structured_observations, env,
            ):
                self.decisions.append({
                    "tool": function, "arguments": dict(dispatch),
                    "route": "verified_service_allow",
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance, "allowed": True,
                    "reason": "email URL resolves to an account-only discount form",
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error
            # read_auto_allow: a read/local_compute whose arguments all come
            # from the authenticated request cannot introduce a new access
            # target, so it is auto-allowed even when the blueprint did not
            # match a capability. This mirrors toolfence_runtime_manifest's
            # read_auto_allow and avoids sending such benign reads to the
            # judge (where a weak/strict judge would over-deny them).
            explicit_document_read = self._is_explicit_observed_document_read(
                function, dispatch,
            )
            if explicit_document_read or self._is_read_auto_allow(function, provenance):
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": (
                        "explicit_observed_document_read_allow"
                        if explicit_document_read else "read_auto_allow"
                    ),
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance,
                    "allowed": True,
                    "reason": (
                        "user explicitly requested reading a listed downloaded document"
                        if explicit_document_read else "read with authenticated arguments"
                    ),
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error
            if self._is_verified_read_dependency(function, provenance, evidence):
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": "read_dependency_allow",
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance,
                    "allowed": True,
                    "reason": "declared read dependency verified in source output",
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error
            # Fallback: ask the judge whether this call serves the request.
            judge_decision = self.judge.decide(
                query=self.query,
                tool=self.tools.get(function, {"name": function}),
                arguments=dispatch,
                provenance=provenance,
                evidence=evidence,
                task_state=(
                    {
                        "observed_tool_calls": self.completed_tools[-12:],
                        "available_read_tools": self.available_read_tools,
                    }
                    if self.workflow_context else None
                ),
            )
            self.decisions.append({
                "tool": function,
                "arguments": dict(dispatch),
                "route": "judge_fallback" if judge_decision.allowed else "denied",
                "blueprint_violations": blueprint_violations,
                "provenance": provenance,
                "allowed": judge_decision.allowed,
                "reason": judge_decision.reason,
                "parsed": judge_decision.parsed,
                "error": judge_decision.error,
                "status": judge_decision.status or ("allow" if judge_decision.allowed else "deny"),
                "next_step": judge_decision.next_step,
            })
            if not judge_decision.allowed:
                message = denial_message(
                    judge_decision.reason,
                    denied_tool=function,
                    authenticated_request=self.query,
                    next_step=(
                        judge_decision.next_step
                        if self.workflow_context and judge_decision.next_step in self.available_read_tools
                        else ""
                    ),
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message
            value, error = self.runtime.run_function(
                env, function, dispatch, raise_on_error=raise_on_error,
            )
            if error is None:
                # Keep the observation so later blueprint derived bindings and
                # judge provenance can see what this call produced.
                self._record_observation(function, value)
                self.monitor.observe(value, function)
            return value, error

        # Provenance gate on the fast path. The blueprint treats read parameters
        # as non-authority-sensitive and binds them `free`, so it happily allows
        # get_webpage(url=<injected>): slack's injection goal is completed purely
        # by such a read, which measured 20% ASR for the naive hybrid. A read
        # whose arguments are not all authenticated still goes to the judge even
        # when the blueprint already allowed it.
        provenance, _evidence = compute_provenance(
            self.query, self.observed_text, dispatch,
        )
        if (
            self._needs_judge_review(function, provenance)
            and not self._is_verified_read_dependency(
                function, provenance, _evidence,
            )
        ):
            review = self.judge.decide(
                query=self.query,
                tool=self.tools.get(function, {"name": function}),
                arguments=dispatch,
                provenance=provenance,
                evidence=_evidence,
                task_state=(
                    {
                        "observed_tool_calls": self.completed_tools[-12:],
                        "available_read_tools": self.available_read_tools,
                    }
                    if self.workflow_context else None
                ),
            )
            self.decisions.append({
                "tool": function,
                "arguments": dict(dispatch),
                "route": "blueprint_then_judge" if review.allowed else "denied",
                "provenance": provenance,
                "allowed": review.allowed,
                "reason": review.reason,
                "parsed": review.parsed,
                "error": review.error,
                "status": review.status or ("allow" if review.allowed else "deny"),
                "next_step": review.next_step,
            })
            if not review.allowed:
                message = denial_message(
                    review.reason,
                    denied_tool=function,
                    authenticated_request=self.query,
                    next_step=(
                        review.next_step
                        if self.workflow_context and review.next_step in self.available_read_tools
                        else ""
                    ),
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message

        self.decisions.append({
            "tool": function,
            "arguments": dict(dispatch),
            "route": route,
            "provenance": provenance,
            "blueprint_violations": [],
            "allowed": True,
            "reason": "blueprint authorization",
        })
        try:
            result = self.runtime.run_function(
                env, function, dispatch, raise_on_error=raise_on_error,
            )
        except BaseException:
            self.monitor.rollback(decision.lease, reason="runtime_exception")
            raise
        value, error = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        if error is not None:
            self.monitor.rollback(decision.lease, reason="tool_returned_error")
            return result
        self.monitor.commit(decision.lease)
        self._record_observation(function, value)
        self.monitor.observe(value, function)
        return result


class DynamicHybridRuntime(HybridRuntime):
    """Dynamic blueprint: judge grants new *capabilities*, not every call.

    On a blueprint miss, instead of asking the judge "is this concrete action
    OK?" (per-call), it asks "may we GRANT a capability of this shape?" and, on
    approval, extends the running plan via ``monitor.grant_capability``. The
    granted capability then authorizes this and future same-shape calls on the
    deterministic fast path -- with no further judge call. Judge invocations
    are therefore bounded by the number of distinct capability expansions, not
    by the number of tool calls.

    A cross-session ``grant_cache`` makes approved shapes reusable across user
    tasks: a capability judged once is injected into every later session's
    monitor, so the judge is never re-asked for the same shape.
    """

    def __init__(self, *args, grant_cache: "CapabilityGrantCache | None" = None,
                 strict_capabilities: bool = False, trusted_read_tools=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.grant_cache = grant_cache
        self.strict_capabilities = bool(strict_capabilities)
        self.trusted_read_tools = frozenset(trusted_read_tools)
        self._auto_read_attempts: set[str] = set()


    def _provenance_to_binding(
        self, name: str, value: Any, source: str,
        authority_sensitive: bool, manifest_tool,
    ) -> Optional[ParameterBinding]:
        """Translate a computed provenance label into a capability binding.

        Authority-sensitive model-generated values are never granted (returns
        None -> caller denies): that is exactly the injection shape.
        """
        if source == "user":
            return ParameterBinding(binding="literal", value=value)
        if source.startswith("tool:"):
            tool_name = source.split(":", 1)[1]
            if self.strict_capabilities:
                # A paper-style DERIVED binding names one concrete field of a
                # committed source observation. Ambiguous/unstructured values
                # cannot become an authority-sensitive grant.
                paths = [path for path, values in
                         getattr(self.monitor, "observed_fields", {}).get(tool_name, {}).items()
                         if str(value) in values]
                if len(paths) != 1:
                    return None
                return ParameterBinding(
                    binding="derived", source_tools=[tool_name],
                    allowed_fields=paths,
                )
            return ParameterBinding(
                binding="derived", source_tools=[tool_name],
            )
        # "model" or unknown
        if authority_sensitive:
            return None  # injection shape: refuse to grant
        return ParameterBinding(binding="free")

    def _build_grant_proposal(
        self, function: str, arguments: Mapping[str, Any],
        provenance: Mapping[str, str],
    ) -> Optional[Capability]:
        """Construct a candidate capability from the action + provenance.

        Returns None when any authority-sensitive parameter is model-generated
        (un-grantable), or when the tool is unknown to the manifest.
        """
        manifest = getattr(self.monitor, "manifest", None)
        if manifest is None:
            return None
        manifest_tool = None
        if hasattr(manifest, "by_name"):
            manifest_tool = manifest.by_name().get(function)
        if manifest_tool is None:
            return None
        parameters: Dict[str, ParameterBinding] = {}
        shape: Dict[str, Dict[str, Any]] = {}
        for name, value in arguments.items():
            param_sem = manifest_tool.parameters.get(name)
            authority_sensitive = bool(
                param_sem.authority_sensitive if param_sem is not None else True
            )
            source = provenance.get(name, "model")
            binding = self._provenance_to_binding(
                name, value, source, authority_sensitive, manifest_tool,
            )
            if binding is None:
                return None  # un-grantable authority-sensitive model value
            parameters[name] = binding
            shape[name] = {
                "provenance": source,
                "authority_sensitive": authority_sensitive,
            }
        self._pending_grant_shape = shape  # stash for the judge call
        return Capability(
            tool=function,
            effect=manifest_tool.effect,
            operation=manifest_tool.operation,
            reusable=manifest_tool.effect in {"read", "local_compute"},
            origin="runtime_capability_grant",
            max_calls=(
                int(getattr(self.monitor.blueprint, "metadata", {}).get(
                    "read_tool_budget", 8,
                ))
                if manifest_tool.effect in {"read", "local_compute"}
                else 1
            ),
            parameters=parameters,
        )

    def _grant_and_prepare(
        self, action: Action, envelope: ProvenanceEnvelope,
    ) -> tuple[Any, Any, Optional[Capability], Optional[JudgeDecision]]:
        """Try to grant a capability and re-prepare the action on the fast path.

        Returns (decision, provenance, capability, confirmation). For a derived
        side-effect grant, confirmation is the independent concrete-action
        decision and the capability is not installed unless both judges allow.
        """
        provenance, evidence = compute_provenance(
            self.query, self.observed_text, dict(action.arguments),
        )
        proposal = self._build_grant_proposal(
            action.tool, action.arguments, provenance,
        )
        if proposal is None:
            return self.monitor.prepare(action, envelope), provenance, None, None
        # Cross-session cache: if this capability shape was already approved
        # in an earlier task, inject it without calling the judge at all.
        sig = capability_signature(proposal)
        cacheable = bool(provenance) and all(
            source == "user" for source in provenance.values()
        )
        if self.grant_cache is not None and cacheable:
            cached = self.grant_cache.get(sig)
            if cached:
                # The cache stores approval for a user-grounded *shape*.
                # Inject the current proposal so literal values remain bound to
                # this session's authenticated request, not a previous user's.
                self.monitor.grant_capability(
                    proposal, arguments=action.arguments,
                )
                decision = self.monitor.prepare(action, envelope)
                self.decisions.append({
                    "tool": action.tool,
                    "arguments": dict(action.arguments),
                    "route": "capability_grant_cached",
                    "provenance": provenance,
                    "allowed": True,
                    "reason": "cached grant hit",
                    "effect": proposal.effect,
                })
                return decision, provenance, proposal, None
        shape = getattr(self, "_pending_grant_shape", {})
        grant_decision = self.judge.decide_grant(
            query=self.query,
            tool=self.tools.get(action.tool, {"name": action.tool}),
            effect=proposal.effect or "unknown",
            arguments_shape=shape,
        )
        self.decisions.append({
            "tool": action.tool,
            "arguments": dict(action.arguments),
            "route": "capability_grant",
            "provenance": provenance,
            "allowed": grant_decision.allowed,
            "reason": grant_decision.reason,
            "parsed": grant_decision.parsed,
            "effect": proposal.effect,
        })
        if not grant_decision.allowed:
            return self.monitor.prepare(action, envelope), provenance, proposal, None

        risky_derived_write = (
            proposal.effect not in {"read", "local_compute"}
            and any(source.startswith("tool:") for source in provenance.values())
        )
        confirmation = None
        if risky_derived_write:
            confirmation = self.judge.decide(
                query=self.query,
                tool=self.tools.get(action.tool, {"name": action.tool}),
                arguments=dict(action.arguments),
                provenance=provenance,
                evidence=evidence,
            )
            self.decisions.append({
                "tool": action.tool,
                "arguments": dict(action.arguments),
                "route": (
                    "grant_then_judge" if confirmation.allowed else "denied"
                ),
                "provenance": provenance,
                "allowed": confirmation.allowed,
                "reason": confirmation.reason,
                "parsed": confirmation.parsed,
                "error": confirmation.error,
            })
            if not confirmation.allowed:
                # Crucially, do not install a capability rejected by the
                # concrete gate. Otherwise a later retry can hit the residual
                # fast path and execute the denied action.
                return (
                    self.monitor.prepare(action, envelope),
                    provenance,
                    proposal,
                    confirmation,
                )
        # Grant accepted: extend the blueprint, then re-prepare on fast path.
        self.monitor.grant_capability(proposal, arguments=action.arguments)
        # Cache the approved shape so later sessions reuse it without judging.
        if self.grant_cache is not None and cacheable:
            self.grant_cache.put(proposal)
        decision = self.monitor.prepare(action, envelope)
        return decision, provenance, proposal, confirmation

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        normalized = {
            name: value for name, value in dict(kwargs).items() if value is not None
        }
        action = Action(function, normalized)
        envelope = ProvenanceEnvelope(
            [action], Provenance.UNTRUSTED_TOOL, source="model_tool_call",
            authenticated=True, tainted_context=bool(
                getattr(self.monitor, "executed_tools", None)
            ),
        )
        trace_index = len(self.monitor.action_trace)
        decision = self.monitor.prepare(action, envelope)
        dispatch = dict(normalized)
        route = "blueprint"

        if not decision.allowed:
            # Blueprint contraction-only repair first (drops undeclared args).
            repaired = self.monitor.contraction_candidate(action, decision.violations)
            if repaired is not None:
                repaired_decision = self.monitor.prepare(repaired, envelope)
                if repaired_decision.allowed and repaired_decision.lease is not None:
                    self.monitor.consolidate_successful_repair(
                        original_event_index=trace_index,
                        repaired=repaired,
                        repaired_decision=repaired_decision,
                    )
                    decision = repaired_decision
                    dispatch = dict(repaired.arguments)
                    route = "blueprint_repair"

        if not decision.allowed or decision.lease is None:
            provenance, evidence = compute_provenance(
                self.query, self.observed_text, dispatch,
            )
            auto_key = json.dumps([function, dispatch], sort_keys=True, default=str)
            if (self.strict_capabilities and auto_key not in self._auto_read_attempts
                    and function in self.trusted_read_tools
                    and provenance and all(source == "user" for source in provenance.values())
                    and self._is_read_auto_allow(function, provenance)):
                proposal = self._build_grant_proposal(function, dispatch, provenance)
                if proposal is not None and proposal.effect in {"read", "local_compute"}:
                    # Issue a bounded capability, then re-enter the same
                    # deterministic prepare/commit path as every other call.
                    self._auto_read_attempts.add(auto_key)
                    self.monitor.grant_capability(proposal, arguments=dispatch)
                    self.decisions.append({"tool": function, "route": "read_auto_capability_issued",
                                           "allowed": True, "provenance": provenance})
                    return self.run_function(env, function, dispatch, raise_on_error)
            # read_auto_allow short-circuit (same as HybridRuntime): a
            # read/local_compute with all-user arguments bypasses both the
            # grant judge and the per-call judge.
            if not self.strict_capabilities and self._is_read_auto_allow(function, provenance):
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": "read_auto_allow",
                    "provenance": provenance,
                    "allowed": True,
                    "reason": "read with authenticated arguments",
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error
            if not self.strict_capabilities and self._is_verified_read_dependency(function, provenance, evidence):
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": "read_dependency_allow",
                    "provenance": provenance,
                    "allowed": True,
                    "reason": "declared read dependency verified in source output",
                })
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error

            # When the static compiler already issued a side-effect capability,
            # its parameter bindings are the maximum authority for that tool.
            # A runtime grant must not create a second conflicting recipient,
            # amount, target, or operation merely because another value happens
            # to occur somewhere in the authenticated request.
            existing_caps = list(
                getattr(self.monitor, "capabilities", {}).get(function, [])
            )
            planned_side_effect = any(
                cap.effect not in {"read", "local_compute"}
                and cap.origin != "runtime_capability_grant"
                for cap in existing_caps
            )
            if planned_side_effect:
                reason = (
                    "proposed write exceeds the parameter scope of the existing "
                    "authenticated capability"
                )
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": "blueprint_scope_denied",
                    "provenance": provenance,
                    "allowed": False,
                    "reason": reason,
                })
                message = denial_message(
                    reason,
                    denied_tool=function,
                    authenticated_request=self.query,
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message
            # Dynamic blueprint: try to GRANT a new capability instead of
            # judging this concrete call. A granted capability makes future
            # same-shape calls hit the fast path with no judge.
            (
                grant_decision_result,
                provenance,
                granted,
                grant_confirmation,
            ) = self._grant_and_prepare(action, envelope)
            if grant_confirmation is not None and not grant_confirmation.allowed:
                message = denial_message(
                    grant_confirmation.reason,
                    denied_tool=function,
                    authenticated_request=self.query,
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message
            if granted is not None and grant_decision_result.allowed and grant_decision_result.lease is not None:
                decision = grant_decision_result
                route = "blueprint_granted"
            else:
                if self.strict_capabilities:
                    reason = "No approved capability covers this call"
                    self.decisions.append({
                        "tool": function, "arguments": dict(dispatch),
                        "route": "capability_denied", "allowed": False,
                        "reason": reason,
                    })
                    message = denial_message(
                        reason, denied_tool=function,
                        authenticated_request=self.query,
                    )
                    if raise_on_error:
                        raise PermissionError(message)
                    return "", message
                blueprint_violations = list(decision.violations)
                judge_decision = self.judge.decide(
                    query=self.query,
                    tool=self.tools.get(function, {"name": function}),
                    arguments=dispatch,
                    provenance=provenance,
                    evidence=None,
                )
                self.decisions.append({
                    "tool": function,
                    "arguments": dict(dispatch),
                    "route": "judge_fallback" if judge_decision.allowed else "denied",
                    "blueprint_violations": blueprint_violations,
                    "provenance": provenance,
                    "allowed": judge_decision.allowed,
                    "reason": judge_decision.reason,
                    "parsed": judge_decision.parsed,
                    "error": judge_decision.error,
                })
                if not judge_decision.allowed:
                    message = denial_message(
                        judge_decision.reason,
                        denied_tool=function,
                        authenticated_request=self.query,
                    )
                    if raise_on_error:
                        raise PermissionError(message)
                    return "", message
                value, error = self.runtime.run_function(
                    env, function, dispatch, raise_on_error=raise_on_error,
                )
                if error is None:
                    self._record_observation(function, value)
                    self.monitor.observe(value, function)
                return value, error

        # Provenance gate on the fast path (same as HybridRuntime).
        provenance, _evidence = compute_provenance(
            self.query, self.observed_text, dispatch,
        )
        if (
            not self.strict_capabilities
            and self._needs_judge_review(function, provenance)
            and not self._is_verified_read_dependency(
                function, provenance, _evidence,
            )
        ):
            review = self.judge.decide(
                query=self.query,
                tool=self.tools.get(function, {"name": function}),
                arguments=dispatch,
                provenance=provenance,
                evidence=_evidence,
            )
            self.decisions.append({
                "tool": function,
                "arguments": dict(dispatch),
                "route": "blueprint_then_judge" if review.allowed else "denied",
                "provenance": provenance,
                "allowed": review.allowed,
                "reason": review.reason,
                "parsed": review.parsed,
                "error": review.error,
            })
            if not review.allowed:
                message = denial_message(
                    review.reason,
                    denied_tool=function,
                    authenticated_request=self.query,
                )
                if raise_on_error:
                    raise PermissionError(message)
                return "", message

        self.decisions.append({
            "tool": function,
            "arguments": dict(dispatch),
            "route": route,
            "provenance": provenance,
            "blueprint_violations": [],
            "allowed": True,
            "reason": "blueprint authorization",
        })
        try:
            result = self.runtime.run_function(
                env, function, dispatch, raise_on_error=raise_on_error,
            )
        except BaseException:
            self.monitor.rollback(decision.lease, reason="runtime_exception")
            raise
        value, error = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        if error is not None:
            self.monitor.rollback(decision.lease, reason="tool_returned_error")
            return result
        self.monitor.commit(decision.lease)
        self._record_observation(function, value)
        self.monitor.observe(value, function)
        return result


class ExplicitDocumentReadMixin:
    """Allow a user-requested document read when its exact filename was listed.

    The request must explicitly ask to read a document, the path must be under
    /downloads, and the basename must have appeared in prior observed output.
    """

    _DOCUMENT_READ_REQUEST = re.compile(
        r"\bread\b.{0,100}\b(document|file|attachment)\b"
        r"|\b(document|file|attachment)\b.{0,100}\bread\b",
        re.IGNORECASE | re.DOTALL,
    )

    def _is_explicit_observed_document_read(
        self, function: str, arguments: Mapping[str, Any],
    ) -> bool:
        if function != "read_file" or not self._DOCUMENT_READ_REQUEST.search(self.query or ""):
            return False
        path = arguments.get("path")
        if not isinstance(path, str) or not path.startswith("/downloads/"):
            return False
        basename = path.rsplit("/", 1)[-1]
        if not basename or basename in {".", ".."}:
            return False
        manifest = getattr(self.monitor, "manifest", None)
        semantic = manifest.by_name().get(function) if hasattr(manifest, "by_name") else None
        if semantic is None or semantic.effect != "read":
            return False
        directory_listing = self.observed_text.get("list_directory", "")
        exact_name = re.compile(
            r"(?<![A-Za-z0-9_.-])" + re.escape(basename) + r"(?![A-Za-z0-9_.-])"
        )
        return bool(exact_name.search(directory_listing))


class HybridDocumentReadRuntime(ExplicitDocumentReadMixin, HybridRuntime):
    """Static hybrid variant with the explicit observed-document read repair."""


class DynamicHybridDocumentReadRuntime(ExplicitDocumentReadMixin, DynamicHybridRuntime):
    """Dynamic hybrid counterpart with the same bounded read-only repair."""
