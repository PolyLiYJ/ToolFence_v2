"""AgentDojo integration for ToolFence."""

from .hybrid.agentdojo_defense import HybridAuthorizationBoundary
from .colored_runtime import TransactionalColoredToolFenceRuntime
from .guarded_net import GuardedTransactionNet


class ToolFenceAuthorizationBoundary(HybridAuthorizationBoundary):
    name = "toolfence_authorization_boundary"

    def __init__(self, compiler, judge, runtime_class=None, *, trusted_catalog,
                 transaction_net, host_bindings, host_actor, host_reviewer,
                 policy_revision, trusted_read_tools=(), **kwargs):
        # This paper profile cannot infer authority from agent-facing catalog
        # descriptions or an LLM-generated tool manifest.
        if getattr(compiler, "pinned_manifest", None) is None:
            raise ValueError("ToolFence requires a controller-pinned manifest")
        if not trusted_catalog:
            raise ValueError("ToolFence requires a controller-owned tool catalog")
        if not isinstance(transaction_net, GuardedTransactionNet):
            raise ValueError("ToolFence requires a guarded transactional ledger")
        if host_bindings is None or not host_actor or host_reviewer is None:
            raise ValueError("ToolFence requires trusted adapters, actor and reviewer")
        if not policy_revision:
            raise ValueError("ToolFence requires a pinned policy revision")
        if runtime_class is not None and not issubclass(runtime_class, TransactionalColoredToolFenceRuntime):
            raise ValueError("ToolFence runtime must retain the guarded ledger boundary")
        pinned = compiler.pinned_manifest.by_name()
        for name in trusted_read_tools:
            semantic = pinned.get(name)
            if (semantic is None or not semantic.reviewed_local_read
                    or semantic.effect not in {"read", "local_compute"}
                    or semantic.external_disclosure or semantic.mutates_state):
                raise ValueError("Automatic read requires pinned no-effect review")
        if kwargs.pop("dynamic", True) is not True:
            raise ValueError("ToolFence paper profile requires dynamic grants")
        kwargs["dynamic"] = True
        kwargs["strict_capabilities"] = True
        kwargs["trusted_catalog"] = trusted_catalog
        kwargs["trusted_read_tools"] = trusted_read_tools
        kwargs["policy_revision"] = policy_revision
        kwargs["host_runtime_kwargs"] = {
            "transaction_net": transaction_net,
            "host_bindings": host_bindings,
            "host_actor": host_actor,
            "host_reviewer": host_reviewer,
            "refine_missing": True,
        }
        super().__init__(
            compiler,
            judge,
            runtime_class=runtime_class or TransactionalColoredToolFenceRuntime,
            **kwargs,
        )
