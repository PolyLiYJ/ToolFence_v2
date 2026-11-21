"""AgentDojo boundary for the hybrid (blueprint + runtime judge) defense."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime

from ..judge.judge import RuntimeJudge
from ..judge.runtime import OpenAICompatibleJudgeBackend
from ..semantic import OpenAICompatibleJSONBackend, SemanticPolicyCompiler
from ..semantic.agentdojo_defense import ToolFenceSemanticBoundary
from .grant_cache import CapabilityGrantCache
from .runtime import DynamicHybridRuntime, HybridRuntime


class HybridAuthorizationBoundary(BasePipelineElement):
    name = "hybrid_authorization_boundary"

    def __init__(
        self,
        compiler: SemanticPolicyCompiler,
        judge: RuntimeJudge,
        *,
        verbose: bool = False,
        inject_plan_guidance: bool = False,
        dynamic: bool = False,
        runtime_class: type | None = None,
        workflow_context: bool = False,
        trusted_catalog=None,
        strict_capabilities: bool = False,
        trusted_read_tools=(),
        host_runtime_kwargs=None,
        policy_revision: str = "legacy",
    ):
        super().__init__()
        self.compiler = compiler
        self.judge = judge
        self.verbose = verbose
        # The blueprint's capability-carrying system message is optional here.
        # Tools are visible either way, so the plan message is not needed to
        # avoid starvation, and it measurably distracts small agents.
        self.inject_plan_guidance = bool(inject_plan_guidance)
        # dynamic=True selects DynamicHybridRuntime: on a blueprint miss, the
        # judge is asked to GRANT a new capability (extending the running
        # blueprint) rather than judging each concrete call. Fewer judge calls,
        # bounded by distinct capability expansions.
        self.dynamic = bool(dynamic)
        self.runtime_class = runtime_class
        self.workflow_context = bool(workflow_context)
        self.trusted_catalog = list(trusted_catalog) if trusted_catalog is not None else None
        self.strict_capabilities = bool(strict_capabilities)
        self.trusted_read_tools = frozenset(trusted_read_tools)
        self.host_runtime_kwargs = dict(host_runtime_kwargs or {})
        self.policy_revision = policy_revision
        # Cross-session cache of judge-approved capability shapes. A grant
        # authorized in one user task is reused in later tasks of the same
        # shape, so the judge is not re-asked. Only enabled for dynamic mode.
        self.grant_cache = CapabilityGrantCache(revision=policy_revision) if self.dynamic else None
        self._semantic = ToolFenceSemanticBoundary(compiler, trusted_catalog=self.trusted_catalog)
        self._sessions: List[Dict[str, Any]] = []
        self._counter = 0

    def audit_snapshot(self) -> List[Dict[str, Any]]:
        return list(self._sessions)

    def audit_cursor(self) -> int:
        return len(self._sessions)

    def annotate_since(self, cursor: int, **evaluation: Any) -> List[str]:
        ids = []
        for item in self._sessions[cursor:]:
            item.setdefault("evaluation", {}).update(evaluation)
            ids.append(item["session_id"])
        return ids

    @staticmethod
    def _tool_catalog(runtime: FunctionsRuntime) -> List[Dict[str, Any]]:
        catalog = []
        for function in runtime.functions.values():
            parameters = function.parameters
            if hasattr(parameters, "model_json_schema"):
                schema = parameters.model_json_schema()
            elif hasattr(parameters, "schema"):
                schema = parameters.schema()
            else:
                schema = {"type": "object", "properties": {}}
            catalog.append({
                "type": "function",
                "function": {
                    "name": function.name,
                    "description": function.description,
                    "parameters": schema,
                },
            })
        return catalog

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence = (),
        extra_args: dict = {},
    ):
        if isinstance(runtime, HybridRuntime):
            return query, runtime, env, messages, extra_args
        self._counter += 1
        session_id = f"hybrid-authorization:{self._counter}"
        catalog = (list(self.trusted_catalog) if self.trusted_catalog is not None
                   else self._tool_catalog(runtime))
        # Compile the blueprint through the v2 boundary so its fail-closed
        # handling stays identical, then reuse its monitor as the fast path.
        _, semantic_runtime, env, boundary_messages, _ = self._semantic.query(
            query, runtime, env, messages, dict(extra_args),
        )
        monitor = semantic_runtime.monitor
        # Cached entries are approval markers for user-grounded shapes, not
        # concrete cross-user authority. DynamicHybridRuntime checks the cache
        # against the current proposal and injects a fresh capability carrying
        # this session's authenticated literal values.
        runtime_cls = self.runtime_class or (
            DynamicHybridRuntime if self.dynamic else HybridRuntime
        )
        runtime_kwargs = {"grant_cache": self.grant_cache} if self.dynamic else {}
        if self.workflow_context:
            runtime_kwargs["workflow_context"] = True
        if self.strict_capabilities:
            runtime_kwargs["strict_capabilities"] = True
            runtime_kwargs["trusted_read_tools"] = self.trusted_read_tools
        runtime_kwargs.update(self.host_runtime_kwargs)
        guarded = runtime_cls(
            runtime, monitor, self.judge, query=query, tools=catalog,
            **runtime_kwargs,
        )
        self._sessions.append({
            "session_id": session_id,
            "query": query,
            "authorized_tools": sorted(name for name in runtime.functions),
            "blueprint_plan": semantic_runtime.monitor.plan.model_dump(mode="json"),
            "decisions": guarded.decisions,
            "grant_cache_snapshot": (
                self.grant_cache.snapshot() if self.grant_cache is not None else None
            ),
            "evaluation": {},
        })
        metadata = {
            **extra_args,
            "hybrid_authorization_session_id": session_id,
            "hybrid_authorization": True,
        }
        if self.verbose:
            print(f"[HybridAuthorization] {session_id}")
        out_messages = boundary_messages if self.inject_plan_guidance else list(messages)
        return query, guarded, env, out_messages, metadata
