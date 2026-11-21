"""ToolFence variant with deterministic colored-token transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .judge.prompts import denial_message
from .colored_net import ColoredAuthorityNet
from .runtime import ToolFenceRuntime


class ColoredToolFenceRuntime(ToolFenceRuntime):
    def __init__(self, *args, transaction_net=None, host_bindings=None,
                 host_read_guard=None, require_transactions=False,
                 host_actor=None, host_reviewer=None, refine_missing=False, **kwargs):
        if require_transactions and (transaction_net is None or host_bindings is None):
            raise ValueError("not_evaluated: trusted transaction contracts are required")
        super().__init__(*args, **kwargs)
        if transaction_net is not None:
            from .transaction_net import TransactionBoundary
            from .guarded_net import GuardedTransactionNet, GuardedBoundary
            if isinstance(transaction_net, GuardedTransactionNet):
                if not host_actor or host_reviewer is None:
                    raise ValueError("Guarded net requires host actor and activation reviewer")
                self.transaction_boundary = GuardedBoundary(
                    self.runtime, transaction_net, host_bindings or {}, host_read_guard,
                    actor=host_actor, reviewer=host_reviewer,
                    refine_missing=refine_missing)
            else:
                self.transaction_boundary = TransactionBoundary(
                    self.runtime, transaction_net, host_bindings or {}, host_read_guard)
            self.runtime = self.transaction_boundary
        self.authority_net = ColoredAuthorityNet(self.query)

    def run_function(
        self, env: Any, function: str, kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        arguments = {key: value for key, value in dict(kwargs).items() if value is not None}
        decision = self.authority_net.decide(function, arguments)
        if decision is not None:
            _, reason, next_step = decision
            self.decisions.append({
                "tool": function, "arguments": arguments, "route": "colored_net_deny",
                "allowed": False, "reason": reason,
                "authority_marking": self.authority_net.snapshot(),
            })
            message = denial_message(
                reason, denied_tool=function,
                authenticated_request=self.query, next_step=next_step,
            )
            if raise_on_error:
                raise PermissionError(message)
            return "", message
        before = len(self.decisions)
        result = super().run_function(env, function, arguments, raise_on_error)
        _, error = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        if error is None and any(d.get("allowed") for d in self.decisions[before:]):
            self.authority_net.observe(function, arguments)
        return result


class TransactionalColoredToolFenceRuntime(ColoredToolFenceRuntime):
    """Evaluation entry point requiring actual host contracts and ledger dispatch."""
    def __init__(self, *args, **kwargs):
        kwargs["require_transactions"] = True
        super().__init__(*args, **kwargs)
