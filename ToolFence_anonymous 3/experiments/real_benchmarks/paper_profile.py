from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from toolfence.agentdojo_defense import ToolFenceAuthorizationBoundary
from toolfence.colored_runtime import TransactionalColoredToolFenceRuntime
from toolfence.guarded_net import GuardedTransactionNet, TransitionRule
from toolfence.judge.judge import RuntimeJudge
from toolfence.semantic.architect import SemanticPolicyCompiler
from toolfence.semantic.schema import ParameterSemantic, ToolManifest, ToolSemantic
from toolfence.transaction_net import (
    ADAPTERS,
    HostToolBinding,
    ObjectRef,
    Receipt,
    TaskContract,
    TransactionNet,
)

from . import config


class ReviewedAdapter:
    """Exact reviewed argument adapter without name-based type heuristics."""

    def __init__(self, tool: str, fields: frozenset[str]):
        self.tool = tool
        self.fields = frozenset(fields)
        self.evidence_kind = None
        self.cost_field = None

    def validate(self, bindings: Mapping[str, Any], arguments: Mapping[str, Any]) -> int:
        if set(bindings) != self.fields or set(arguments) != self.fields:
            raise PermissionError("Exact reviewed argument set required")
        if any(bindings[key] != arguments[key] for key in self.fields):
            raise PermissionError("Invocation differs from reviewed argument bindings")
        return 1


OPERATIONAL_PARAMETERS = {
    "limit", "offset", "page", "page_size", "format", "max_results",
    "include_archived", "recursive", "verbose",
}

READ_PREFIXES = (
    "read", "get", "list", "search", "find", "view", "fetch", "query",
    "inspect", "describe", "show", "lookup", "summarize",
)
WRITE_PREFIXES = (
    "write", "create", "update", "edit", "move", "delete", "remove",
    "send", "email", "post", "upload", "transfer", "pay", "purchase",
    "subscribe", "invite", "schedule", "book", "reserve", "set", "add",
)


def _schema_properties(schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    properties = schema.get("properties", {})
    return dict(properties) if isinstance(properties, Mapping) else {}


def classify_tool(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    lowered = name.lower()
    props = _schema_properties(schema)
    parameters = {
        key: {
            "semantic_role": f"schema:{key}:{str(meta.get('type', 'unknown'))}",
            "authority_sensitive": key not in OPERATIONAL_PARAMETERS,
        }
        for key, meta in props.items()
    }
    if any(lowered.startswith(prefix) for prefix in READ_PREFIXES):
        effect, operation = "read", "observe"
        external_disclosure, mutates_state = False, False
    elif any(lowered.startswith(prefix) for prefix in WRITE_PREFIXES):
        if any(token in lowered for token in ("send", "email", "post", "message", "invite")):
            effect, operation = "communication", "communicate"
        elif any(token in lowered for token in ("delete", "remove")):
            effect, operation = "delete", "delete"
        elif any(token in lowered for token in ("transfer", "pay", "purchase", "book")):
            effect, operation = "financial", "financial"
        else:
            effect, operation = "external_write", "update"
        external_disclosure, mutates_state = True, True
    else:
        # Conservative host review fallback: never claim a benign read.
        effect, operation = "external_write", "update"
        external_disclosure, mutates_state = True, True
    return {
        "name": name,
        "effect": effect,
        "operation": operation,
        "resource": "benchmark-tool",
        "reversible": False,
        "reviewed_local_read": False,
        "external_disclosure": external_disclosure,
        "mutates_state": mutates_state,
        "parameters": parameters,
    }


def build_pinned_manifest(catalog: Sequence[Mapping[str, Any]]) -> ToolManifest:
    reviewed = []
    seen: set[str] = set()
    for item in catalog:
        fn = item.get("function", item)
        name = str(fn.get("name") or "")
        if not name or name in seen:
            raise ValueError(f"Invalid or duplicate trusted tool name: {name!r}")
        seen.add(name)
        cls = classify_tool(name, fn.get("parameters", {}))
        reviewed.append(ToolSemantic(
            name=name,
            effect=cls["effect"],
            operation=cls["operation"],
            resource=cls["resource"],
            reversible=cls["reversible"],
            reviewed_local_read=cls["reviewed_local_read"],
            external_disclosure=cls["external_disclosure"],
            mutates_state=cls["mutates_state"],
            parameters={
                key: ParameterSemantic(
                    semantic_role=value["semantic_role"],
                    authority_sensitive=value["authority_sensitive"],
                )
                for key, value in cls["parameters"].items()
            },
        ))
    if not reviewed:
        raise ValueError("Trusted catalog is empty")
    return ToolManifest(tools=reviewed)


def build_pinned_manifest_from_reviewed(semantics: Mapping[str, Mapping[str, Any]]) -> ToolManifest:
    reviewed = []
    for name, spec in semantics.items():
        parameters = {
            key: ParameterSemantic(
                semantic_role=str(value.get("semantic_role", f"reviewed:{key}")),
                authority_sensitive=bool(value.get("authority_sensitive", True)),
            )
            for key, value in (spec.get("parameters") or {}).items()
        }
        reviewed.append(ToolSemantic(
            name=name,
            effect=str(spec["effect"]),
            operation=str(spec["operation"]),
            resource=str(spec.get("resource", "benchmark-resource")),
            reversible=bool(spec.get("reversible", False)),
            reviewed_local_read=bool(spec.get("reviewed_local_read", False)),
            external_disclosure=bool(spec.get("external_disclosure", False)),
            mutates_state=bool(spec.get("mutates_state", False)),
            parameters=parameters,
        ))
    if not reviewed:
        raise ValueError("Reviewed manifest is empty")
    return ToolManifest(tools=reviewed)


def canonical_for_call(tool: str, arguments: Mapping[str, Any], revision: str) -> dict[str, str]:
    digest = hashlib.sha256(TransactionNet.canonical(arguments).encode()).hexdigest()
    ref = ObjectRef("benchmark", "tool", tool, revision).encode()
    return {"operation_ref": ref, "arguments_digest": digest}


@dataclass
class PaperProfileBundle:
    boundary: ToolFenceAuthorizationBoundary
    compiler: SemanticPolicyCompiler
    judge: RuntimeJudge
    backends: Any
    transaction_net: GuardedTransactionNet
    trusted_catalog: list[dict[str, Any]]
    pinned_manifest: ToolManifest
    audit: dict[str, Any]


def build_paper_profile(
    backends: Any,
    tools: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str,
    policy_revision: str | None = None,
    host_actor: str | None = None,
    reviewed_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_audit: Mapping[str, Any] | None = None,
) -> PaperProfileBundle:
    revision = policy_revision or config.POLICY_REVISION
    actor = host_actor or config.HOST_ACTOR
    catalog = [dict(item) for item in tools]
    if reviewed_semantics is None:
        manifest = build_pinned_manifest(catalog)
        classification_source = "deprecated_name_heuristic"
    else:
        manifest = build_pinned_manifest_from_reviewed(reviewed_semantics)
        classification_source = "reviewed_json_manifest"
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    net = GuardedTransactionNet(ledger_path)
    lock = threading.Lock()
    issued: dict[str, dict[str, Any]] = {}

    def receipt(operation: str, result: Any) -> Receipt:
        if isinstance(result, tuple) and len(result) == 2 and result[1] is not None:
            return Receipt(operation, "unknown")
        return Receipt(operation, "committed")

    def reviewer(proposal: Mapping[str, Any]) -> bool:
        task = str(proposal.get("task", ""))
        with lock:
            record = issued.get(task)
        arguments = proposal.get("raw_arguments", proposal.get("arguments", {}))
        return bool(
            record
            and str(proposal.get("tool", "")).startswith("reviewed.")
            and record["adapter"] == proposal.get("tool")
            and dict(arguments) == record["arguments"]
            and record["revision"] == revision
        )

    def issue_contract(tool: str, arguments: Mapping[str, Any]) -> tuple[str, HostToolBinding]:
        args = dict(arguments)
        canonical = canonical_for_call(tool, args, revision)
        digest = canonical["arguments_digest"]
        adapter = f"reviewed.{tool}.{digest}"
        key = f"{tool}:{digest}"
        with lock:
            existing = issued.get(key)
        if existing is not None:
            task = existing["task"]
        else:
            task = str(uuid4())
            budget = str(uuid4())
            ADAPTERS[adapter] = ReviewedAdapter(adapter, frozenset(args.keys()))
            net.issue(TaskContract(task, budget, adapter, args))
            net.configure(
                task,
                initial_stage="ready",
                rules={adapter: TransitionRule("ready", "done")},
                revision=revision,
                expires=time.time() + 600.0,
            )
            with lock:
                record = {
                    "task": task, "adapter": adapter, "arguments": args, "revision": revision,
                }
                issued[key] = record
                issued[task] = record
        binding = HostToolBinding(
            task,
            adapter,
            lambda raw, args=args: dict(raw),
            receipt,
            frozenset(args),
        )
        return task, binding

    class BenchmarkTransactionalRuntime(TransactionalColoredToolFenceRuntime):
        name = "benchmark_toolfence_strict_runtime"

        def run_function(self, env, function, kwargs, raise_on_error=False):
            task, binding = issue_contract(function, kwargs)
            self.runtime.bindings[function] = binding
            return super().run_function(env, function, kwargs, raise_on_error=raise_on_error)

    compiler = SemanticPolicyCompiler(
        backends.compiler,
        pinned_manifest=manifest,
        max_compile_attempts=2,
        completeness_second_pass=True,
    )
    judge = RuntimeJudge(backends.judge, fail_closed=True)
    boundary = ToolFenceAuthorizationBoundary(
        compiler,
        judge,
        runtime_class=BenchmarkTransactionalRuntime,
        trusted_catalog=catalog,
        transaction_net=net,
        host_bindings={},
        host_actor=actor,
        host_reviewer=reviewer,
        policy_revision=revision,
        trusted_read_tools=(),
        inject_plan_guidance=False,
    )
    audit = {
        "profile": "toolfence_paper_strict_adapted",
        "policy_revision": revision,
        "host_actor": actor,
        "adapter": "reviewed.<tool>.<sha256>",
        "adapter_scope": "dynamic_reviewed_exact_argument_adapter",
        "classification_source": classification_source,
        "manifest_audit": dict(manifest_audit or {}),
        "trusted_catalog": [
            {"name": tool["function"]["name"], "classification": (
                reviewed_semantics.get(tool["function"]["name"], {}) if reviewed_semantics is not None
                else classify_tool(tool["function"]["name"], tool["function"].get("parameters", {}))
            )}
            for tool in catalog
        ],
    }
    return PaperProfileBundle(
        boundary=boundary,
        compiler=compiler,
        judge=judge,
        backends=backends,
        transaction_net=net,
        trusted_catalog=catalog,
        pinned_manifest=manifest,
        audit=audit,
    )
