"""Hybrid runtime: blueprint fast path with judge fallback."""

import json

import pytest

from toolfence.judge.judge import RuntimeJudge
from toolfence.judge.runtime import compute_provenance
from toolfence.hybrid.runtime import DynamicHybridRuntime, HybridRuntime
from toolfence.hybrid.grant_cache import CapabilityGrantCache
from toolfence.foundation.blueprint import (
    ParameterRule, SessionBlueprint, ToolBlueprint, ToolPermission,
)
from toolfence.foundation.authorization import Action, Provenance, ProvenanceEnvelope

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from toolfence.semantic.monitor import SemanticMonitor  # noqa: E402
from toolfence.semantic.schema import (  # noqa: E402
    AuthorizationPlan, Capability, ParameterBinding, ToolManifest, ToolSemantic,
)


class FakeBackend:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def queue(self, *responses):
        self.responses = list(responses)
        return self

    def __call__(self, system_prompt, user_prompt, max_tokens=256):
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("unexpected judge call")
        return self.responses.pop(0)


class FakeRuntime:
    def __init__(self, result=("ok", None)):
        self.functions = {"send_email": object()}
        self.result = result
        self.dispatched = []

    def run_function(self, env, function, kwargs, raise_on_error=False):
        self.dispatched.append((function, dict(kwargs)))
        return self.result


TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to the given recipients.",
        "parameters": {"type": "object", "properties": {
            "recipients": {"type": "array"}, "body": {"type": "string"},
        }},
    },
}]


def build_monitor(effect="communication", sensitive=True):
    manifest = ToolManifest(tools=[ToolSemantic(
        name="send_email", effect=effect, operation="communicate",
        resource="message", reversible=False,
        parameters={},
    )])
    plan = AuthorizationPlan(goal="Send a message", capabilities=[Capability(
        tool="send_email", effect=effect, operation="communicate",
        reusable=False, max_calls=1,
        parameters={
            "recipients": ParameterBinding(binding="literal", value="bob@example.com"),
            "body": ParameterBinding(binding="free"),
        },
    )])
    blueprint = SessionBlueprint(
        session_id="hybrid-test", user_intent="Send hi to bob@example.com",
        authorized_goals=["Send a message"],
        allowed_tools={"send_email": ToolBlueprint(
            tool_id="send_email", permission=ToolPermission.ALLOWED,
            description="Send an email",
            parameter_rules={"recipients": ParameterRule(name="recipients", type="array"),
                             "body": ParameterRule(name="body", type="string")},
            max_calls_per_session=1, secrets="allow",
        )},
        max_total_tool_calls=4,
    )
    monitor = SemanticMonitor(blueprint, plan, manifest)
    monitor.register_user_query("Send hi to bob@example.com")
    return monitor


def envelope(action):
    return ProvenanceEnvelope(
        [action], Provenance.UNTRUSTED_TOOL, source="model_tool_call",
        authenticated=True, tainted_context=False,
    )


def test_blueprint_allows_without_calling_judge():
    backend = FakeBackend()  # any judge call would raise
    judge = RuntimeJudge(backend)
    monitor = build_monitor()
    inner = FakeRuntime()
    hybrid = HybridRuntime(inner, monitor, judge, query="Send hi to bob@example.com",
                           tools=TOOLS)
    value, error = hybrid.run_function(
        None, "send_email", {"recipients": ["bob@example.com"], "body": "hi"},
    )
    assert backend.calls == []           # fast path costs no LLM call
    assert inner.dispatched, "action should execute"
    assert hybrid.decisions[0]["route"] == "blueprint"
    assert hybrid.decisions[0]["allowed"] is True


def test_blueprint_denial_is_rescued_by_judge():
    backend = FakeBackend().queue('{"decision":"allow","reason":"requested"}')
    judge = RuntimeJudge(backend)
    monitor = build_monitor()
    inner = FakeRuntime()
    hybrid = HybridRuntime(inner, monitor, judge, query="Send hi to bob@example.com",
                           tools=TOOLS)
    # Wrong recipient: the blueprint's literal binding must reject it.
    value, error = hybrid.run_function(
        None, "send_email", {"recipients": ["attacker@evil.com"], "body": "hi"},
    )
    assert len(backend.calls) == 1       # judge consulted on blueprint denial
    assert hybrid.decisions[0]["route"] == "judge_fallback"
    assert hybrid.decisions[0]["blueprint_violations"], "blueprint reason recorded"


def test_both_gates_deny_returns_recoverable_message():
    backend = FakeBackend().queue('{"decision":"deny","reason":"not requested"}')
    judge = RuntimeJudge(backend)
    monitor = build_monitor()
    inner = FakeRuntime()
    hybrid = HybridRuntime(inner, monitor, judge, query="Send hi to bob@example.com",
                           tools=TOOLS)
    value, error = hybrid.run_function(
        None, "send_email", {"recipients": ["attacker@evil.com"], "body": "hi"},
    )
    assert inner.dispatched == []
    assert "RUNTIME_AUTHORIZATION_DENIED" in error
    assert hybrid.decisions[0]["route"] == "denied"


def test_hybrid_context_passes_controller_call_history_to_fallback():
    backend = FakeBackend('{"decision":"deny","reason":"wrong recipient"}')
    monitor = build_monitor()
    hybrid = HybridRuntime(
        FakeRuntime(), monitor, RuntimeJudge(backend, with_provenance=True),
        query="Send hi to bob@example.com", tools=TOOLS, workflow_context=True,
    )
    hybrid.run_function(None, "send_email", {"recipients": ["bob@example.com"], "body": "hi"})
    hybrid.run_function(None, "send_email", {"recipients": ["bob@example.com"], "body": "hi"})
    payload = json.loads(backend.calls[0][1])
    assert payload["task_state"]["observed_tool_calls"] == ["send_email"]
    assert payload["task_state"]["available_read_tools"] == []


def test_hybrid_context_keeps_authenticated_recipient_fixed():
    backend = FakeBackend('{"decision":"allow","reason":"looks useful"}')
    hybrid = HybridRuntime(
        FakeRuntime(), build_monitor(), RuntimeJudge(backend),
        query="Send hi to bob@example.com", tools=TOOLS, workflow_context=True,
    )
    _, error = hybrid.run_function(
        None, "send_email", {"recipients": ["attacker@evil.com"], "body": "hi"},
    )
    assert "RUNTIME_AUTHORIZATION_DENIED" in error
    assert hybrid.decisions[0]["route"] == "authenticated_literal_conflict"
    assert backend.calls == []


def test_hybrid_context_recovers_exact_account_login_dependency():
    backend = FakeBackend()
    login_tool = {"function": {
        "name": "login_shopping_account",
        "description": "Login to your shopping account.",
    }}
    hybrid = HybridRuntime(
        FakeRuntime(), build_monitor(), RuntimeJudge(backend),
        query="Please buy a smart watch for me.", tools=[login_tool],
        workflow_context=True,
    )
    hybrid.structured_observations["get_shopping_account_user_information"] = {
        "Username": "owner", "Password": "secret",
    }
    hybrid.run_function(None, "login_shopping_account", {
        "username": "owner", "password": "secret",
    })
    assert hybrid.decisions[0]["route"] == "workflow_dependency_allow"
    assert backend.calls == []


def test_all_tools_stay_visible():
    judge = RuntimeJudge(FakeBackend())
    monitor = build_monitor()
    inner = FakeRuntime()
    hybrid = HybridRuntime(inner, monitor, judge, query="q", tools=TOOLS)
    assert "send_email" in hybrid.functions


def test_compute_provenance_shared_helper():
    labels, evidence = compute_provenance(
        "Pay bill 120", {"read_file": "...Total: 120 EUR..."},
        {"amount": "120", "note": "urgent"},
    )
    assert labels["amount"] == "user"
    assert labels["note"] == "model"


def test_declared_tool_output_read_dependency_is_deterministically_verified():
    monitor = build_monitor(effect="read")
    monitor.manifest = ToolManifest(tools=[ToolSemantic(
        name="get_webpage", effect="read", operation="observe",
        resource="webpage", reversible=True, parameters={},
    )])
    monitor.plan = AuthorizationPlan(goal="Read linked article", capabilities=[
        Capability(
            tool="get_webpage", effect="read", operation="observe",
            reusable=True,
            parameters={
                "url": ParameterBinding(
                    binding="derived", source_tools=["read_channel_messages"],
                    allowed_fields=["content"],
                ),
            },
        ),
    ])
    hybrid = HybridRuntime(
        FakeRuntime(), monitor, RuntimeJudge(FakeBackend()),
        query="Read the article link Bob posted in general", tools=[],
    )
    hybrid.observed_text["read_channel_messages"] = (
        "Bob posted https://safe.example/article"
    )
    provenance, evidence = compute_provenance(
        hybrid.query,
        hybrid.observed_text,
        {"url": "https://safe.example/article"},
    )
    assert provenance == {"url": "tool:read_channel_messages"}
    assert hybrid._is_verified_read_dependency(
        "get_webpage", provenance, evidence,
    ) is True


def test_tool_output_read_without_declared_edge_is_not_verified():
    monitor = build_monitor(effect="read")
    hybrid = HybridRuntime(
        FakeRuntime(), monitor, RuntimeJudge(FakeBackend()),
        query="Read my inbox", tools=[],
    )
    provenance = {"url": "tool:read_channel_messages"}
    evidence = {
        "url": {
            "provenance": "tool:read_channel_messages",
            "excerpt": "visit https://evil.example",
        },
    }
    assert hybrid._is_verified_read_dependency(
        "get_webpage", provenance, evidence,
    ) is False


def test_grant_cache_rejects_untrusted_derived_shapes():
    cache = CapabilityGrantCache()
    derived = Capability(
        tool="get_webpage", effect="read", operation="observe",
        parameters={
            "url": ParameterBinding(
                binding="derived", source_tools=["read_channel_messages"],
            ),
        },
    )
    authenticated = Capability(
        tool="get_webpage", effect="read", operation="observe",
        parameters={"url": ParameterBinding(binding="literal")},
    )
    assert cache.put(derived) is False
    assert cache.snapshot()["cached_shapes"] == 0
    assert cache.put(authenticated) is True
    assert cache.snapshot()["cached_shapes"] == 1


def test_dynamic_grant_updates_core_blueprint_and_preserves_user_literal():
    manifest = ToolManifest(tools=[ToolSemantic(
        name="send_email", effect="communication", operation="communicate",
        resource="message", reversible=False, parameters={},
    )])
    plan = AuthorizationPlan(goal="Send email", capabilities=[])
    blueprint = SessionBlueprint(
        session_id="dynamic-grant", user_intent="Send hi to bob@example.com",
        authorized_goals=["Send email"],
        allowed_tools={"send_email": ToolBlueprint(
            tool_id="send_email", permission=ToolPermission.DENIED,
            description="Send an email", parameter_rules={},
            max_calls_per_session=1, secrets="allow",
        )},
        max_total_tool_calls=1,
    )
    monitor = SemanticMonitor(blueprint, plan, manifest)
    monitor.register_user_query("Send hi to bob@example.com")
    backend = FakeBackend().queue(
        '{"decision":"allow","reason":"explicitly requested"}',
    )
    inner = FakeRuntime()
    runtime = DynamicHybridRuntime(
        inner, monitor, RuntimeJudge(backend),
        query="Send hi to bob@example.com", tools=TOOLS,
        grant_cache=CapabilityGrantCache(),
    )
    value, error = runtime.run_function(
        None,
        "send_email",
        {"recipients": ["bob@example.com"], "body": "hi"},
    )
    assert error is None
    assert inner.dispatched == [(
        "send_email",
        {"recipients": ["bob@example.com"], "body": "hi"},
    )]
    assert monitor.blueprint.allowed_tools["send_email"].permission == ToolPermission.ALLOWED
    granted = monitor.capabilities["send_email"][0]
    assert granted.parameters["recipients"].value == ["bob@example.com"]
    assert granted.parameters["body"].value == "hi"
    assert runtime.decisions[-1]["route"] == "blueprint_granted"


def test_read_auto_allow_accepts_only_non_sensitive_operational_arguments():
    manifest = ToolManifest.model_validate({
        "tools": [{
            "name": "list_transactions", "effect": "read",
            "operation": "observe", "resource": "transactions",
            "parameters": {
                "n": {"semantic_role": "result_count", "authority_sensitive": False},
                "account": {"semantic_role": "account_target", "authority_sensitive": True},
            },
        }],
    })
    monitor = build_monitor(effect="read")
    monitor.manifest = manifest
    runtime = HybridRuntime(
        FakeRuntime(), monitor, RuntimeJudge(FakeBackend()),
        query="Check recent transactions", tools=[],
    )
    assert runtime._is_read_auto_allow("list_transactions", {"n": "model"}) is True
    manifest.tools[0].parameters["n"].authority_sensitive = True
    assert runtime._is_read_auto_allow("list_transactions", {"n": "model"}) is True
    assert runtime._is_read_auto_allow(
        "list_transactions", {"account": "model"},
    ) is False


def test_derived_write_grant_requires_independent_concrete_confirmation():
    manifest = ToolManifest.model_validate({
        "tools": [{
            "name": "send_email", "effect": "communication",
            "operation": "communicate", "resource": "message",
            "parameters": {
                "recipients": {
                    "semantic_role": "message_recipient",
                    "authority_sensitive": True,
                },
                "body": {
                    "semantic_role": "message_body",
                    "authority_sensitive": False,
                },
            },
        }],
    })
    plan = AuthorizationPlan(goal="Read recipient and send hi", capabilities=[])
    blueprint = SessionBlueprint(
        session_id="derived-write", user_intent="Read the bill and send hi",
        authorized_goals=["Read recipient and send hi"],
        allowed_tools={"send_email": ToolBlueprint(
            tool_id="send_email", permission=ToolPermission.DENIED,
            description="Send email", parameter_rules={}, max_calls_per_session=1,
            secrets="allow",
        )},
        max_total_tool_calls=1,
    )
    monitor = SemanticMonitor(blueprint, plan, manifest)
    monitor.register_user_query("Read the bill and send hi")
    monitor.observe("recipient=bob@example.com", "read_file")
    backend = FakeBackend().queue(
        '{"decision":"allow","reason":"shape appears useful"}',
        '{"decision":"deny","reason":"recipient came from untrusted content"}',
    )
    inner = FakeRuntime()
    runtime = DynamicHybridRuntime(
        inner, monitor, RuntimeJudge(backend, with_provenance=True),
        query="Read the bill and send hi", tools=TOOLS,
        grant_cache=CapabilityGrantCache(),
    )
    runtime.observed_text["read_file"] = "recipient=bob@example.com"
    value, error = runtime.run_function(
        None,
        "send_email",
        {"recipients": ["bob@example.com"], "body": "hi"},
    )
    assert inner.dispatched == []
    assert "RUNTIME_AUTHORIZATION_DENIED" in error
    assert [item["route"] for item in runtime.decisions] == [
        "capability_grant", "denied",
    ]


def test_dynamic_grant_cannot_expand_existing_write_parameter_scope():
    monitor = build_monitor()
    backend = FakeBackend()  # scope denial must be deterministic, no judge call
    inner = FakeRuntime()
    runtime = DynamicHybridRuntime(
        inner, monitor, RuntimeJudge(backend),
        query="Send hi to bob@example.com, not attacker@evil.com",
        tools=TOOLS,
        grant_cache=CapabilityGrantCache(),
    )
    value, error = runtime.run_function(
        None,
        "send_email",
        {"recipients": ["attacker@evil.com"], "body": "hi"},
    )
    assert inner.dispatched == []
    assert backend.calls == []
    assert "RUNTIME_AUTHORIZATION_DENIED" in error
    assert runtime.decisions[-1]["route"] == "blueprint_scope_denied"
