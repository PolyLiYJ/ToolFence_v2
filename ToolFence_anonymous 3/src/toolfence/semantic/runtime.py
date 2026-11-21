"""Runtime wrapper for ToolFence-Semantic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from toolfence.foundation.authorization import Action, Provenance, ProvenanceEnvelope

from .monitor import SemanticMonitor


class ToolFenceSemanticRuntime:
    def __init__(self, runtime: Any, monitor: SemanticMonitor):
        self.runtime = runtime
        self.monitor = monitor
        self.blocked_calls: List[dict[str, Any]] = []

    @property
    def functions(self) -> Any:
        allowed = set(self.monitor.blueprint.authorized_tools)
        return {
            name: function for name, function in self.runtime.functions.items()
            if name in allowed
        }

    def register_function(self, function: Any) -> Any:
        return self.runtime.register_function(function)

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        # OpenAI-compatible tool callers commonly serialize an omitted optional
        # field as JSON null. Treat that representation as omission, while
        # preserving null for required fields so schema validation still fails
        # closed. This avoids false denials such as search_emails(sender=None).
        normalized_kwargs = dict(kwargs)
        tool_blueprint = self.monitor.blueprint.allowed_tools.get(function)
        if tool_blueprint is not None:
            for name, value in list(normalized_kwargs.items()):
                rule = tool_blueprint.parameter_rules.get(name)
                if value is None and rule is not None and not rule.required:
                    normalized_kwargs.pop(name)

        action = Action(function, normalized_kwargs)
        envelope = ProvenanceEnvelope(
            [action], Provenance.UNTRUSTED_TOOL, source="model_tool_call",
            authenticated=True, tainted_context=bool(self.monitor.executed_tools),
        )
        trace_index = len(self.monitor.action_trace)
        decision = self.monitor.prepare(action, envelope)
        dispatch_kwargs = dict(normalized_kwargs)
        if not decision.allowed:
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
                    dispatch_kwargs = dict(repaired.arguments)
        if not decision.allowed or decision.lease is None:
            event = {"action": action.to_dict(), "violations": list(decision.violations)}
            self.blocked_calls.append(event)
            message = (
                "TOOLFENCE_SEMANTIC_BLOCKED: "
                + "; ".join(decision.violations)
                + ". Recovery may only contract the action or execute an already-authorized "
                  "read/source dependency; no new tool, recipient, destination, target, amount, "
                  "or side effect may be introduced. For a derived parameter, retry only by "
                  "copying the exact observed value, including spelling and apparent typos."
            )
            if raise_on_error:
                raise PermissionError(message)
            return "", message
        try:
            result = self.runtime.run_function(
                env, function, dispatch_kwargs, raise_on_error=raise_on_error,
            )
        except BaseException:
            self.monitor.rollback(decision.lease, reason="runtime_exception")
            raise
        value, error = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        if error is not None:
            self.monitor.rollback(decision.lease, reason="tool_returned_error")
            return result
        self.monitor.commit(decision.lease)
        self.monitor.observe(value, function)
        return result
