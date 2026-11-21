"""Regressions for paper-profile trust boundaries added to the source snapshot."""

import pytest

from toolfence.hybrid.grant_cache import CapabilityGrantCache
from toolfence.hybrid.runtime import DynamicHybridRuntime
from toolfence.judge.judge import RuntimeJudge
from toolfence.semantic.architect import SemanticPolicyCompiler
from toolfence.semantic.monitor import SemanticMonitor
from toolfence.semantic.schema import (
    AuthorizationPlan, Capability, ParameterBinding, ParameterSemantic, ToolManifest, ToolSemantic,
)
from toolfence.foundation.blueprint import SessionBlueprint, ToolBlueprint, ToolPermission
from tests.unit.test_toolfence_hybrid import FakeBackend, FakeRuntime, TOOLS, build_monitor


def test_cross_session_cache_retains_shape_only():
    cache = CapabilityGrantCache(revision="policy-v1")
    first = Capability(
        tool="send_email", effect="communication", operation="communicate",
        parameters={"recipient": ParameterBinding(binding="literal", value="alice@example.org")},
    )
    assert cache.put(first)
    assert "alice@example.org" not in repr(cache.__dict__)
    second = Capability(
        tool="send_email", effect="communication", operation="communicate",
        parameters={"recipient": ParameterBinding(binding="literal", value="bob@example.org")},
    )
    assert cache.get(("send_email", "communication", (("recipient", "literal", (), ()),)))
    assert cache.put(second)
    assert cache.snapshot()["cached_shapes"] == 1
    assert "bob@example.org" not in repr(cache.__dict__)


def test_declared_source_field_does_not_fall_back_to_other_fields():
    monitor = build_monitor()
    monitor.executed_tools.add("lookup")
    monitor.observed_values["lookup"].add("attacker@example.org")
    monitor.observed_fields["lookup"]["other"].add("attacker@example.org")
    binding = ParameterBinding(binding="derived", source_tools=["lookup"],
                               allowed_fields=["recipient"])
    assert monitor._binding_violation("send_email", "recipient", "attacker@example.org", binding)
    monitor.observed_fields["lookup"]["recipient"].add("alice@example.org")
    assert monitor._binding_violation("send_email", "recipient", "alice@example.org", binding) is None


def test_pinned_manifest_skips_model_classification_and_validates_schema():
    calls = []
    pinned = ToolManifest(tools=[ToolSemantic(
        name="send_email", effect="communication", operation="communicate",
        parameters={"recipient": ParameterSemantic(semantic_role="target", authority_sensitive=True)},
    )])
    compiler = SemanticPolicyCompiler(lambda *a: calls.append(a), pinned_manifest=pinned)
    catalog = [{"name": "send_email", "parameters": {"type": "object",
                 "properties": {"recipient": {"type": "string"}}}}]
    assert compiler.compile_manifest(catalog).tools[0].effect == "communication"
    assert calls == []
    with pytest.raises(ValueError, match="parameter classifications"):
        compiler.compile_manifest([{"name": "send_email", "parameters": {"type": "object",
                                   "properties": {"other": {"type": "string"}}}}])


def test_paper_plan_requires_exact_derived_field():
    manifest = ToolManifest(tools=[ToolSemantic(
        name="open_url", effect="read", operation="observe", resource="web",
        parameters={"url": ParameterSemantic(semantic_role="url", authority_sensitive=True)},
    )])
    tools = [{"name": "open_url", "description": "Open URL", "parameters": {
        "type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"],
    }}]
    plan = AuthorizationPlan(goal="Open a page", capabilities=[Capability(
        tool="open_url", parameters={"url": ParameterBinding(
            binding="derived", source_tools=["lookup"],
        )},
    )])
    with pytest.raises(ValueError, match="one declared source field"):
        SemanticPolicyCompiler._validate_plan(
            "Open the URL returned by lookup", plan, manifest, tools, strict_fields=True,
        )


def test_strict_profile_never_falls_back_to_per_call_allow():
    backend = FakeBackend('{"decision":"allow","reason":"yes"}')
    inner = FakeRuntime()
    runtime = DynamicHybridRuntime(
        inner, build_monitor(), RuntimeJudge(backend),
        query="Send hi to bob@example.com", tools=TOOLS,
        strict_capabilities=True,
    )
    _value, error = runtime.run_function(
        None, "send_email", {"recipients": ["mallory@example.org"], "body": "hi"},
    )
    assert error is not None
    assert inner.dispatched == []
    assert backend.calls == []
    assert runtime.decisions[-1]["route"] in {"blueprint_scope_denied", "capability_denied"}


def test_reviewed_read_issues_capability_before_dispatch():
    tool = "list_transactions"
    manifest = ToolManifest(tools=[ToolSemantic(
        name=tool, effect="read", operation="observe", reviewed_local_read=True,
        parameters={"n": ParameterSemantic(semantic_role="result_count", authority_sensitive=False)},
    )])
    blueprint = SessionBlueprint(
        session_id="read-auto", user_intent="List 3 transactions",
        allowed_tools={tool: ToolBlueprint(
            tool_id=tool, permission=ToolPermission.DENIED, description=tool,
            parameter_rules={}, max_calls_per_session=1, secrets="allow",
        )}, max_total_tool_calls=3,
    )
    monitor = SemanticMonitor(blueprint, AuthorizationPlan(goal="List transactions"), manifest)
    monitor.register_user_query("List 3 transactions")
    inner = FakeRuntime()
    backend = FakeBackend()
    runtime = DynamicHybridRuntime(
        inner, monitor, RuntimeJudge(backend), query="List 3 transactions",
        tools=[{"type": "function", "function": {"name": tool, "description": "",
                "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}}],
        strict_capabilities=True, trusted_read_tools={tool},
    )
    _value, error = runtime.run_function(None, tool, {"n": 3})
    assert error is None
    assert inner.dispatched == [(tool, {"n": 3})]
    assert backend.calls == []
    assert monitor.capabilities[tool]
    assert any(item["route"] == "read_auto_capability_issued" for item in runtime.decisions)
