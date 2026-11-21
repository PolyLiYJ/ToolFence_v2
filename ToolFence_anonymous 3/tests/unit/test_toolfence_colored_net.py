"""Colored-token authorization: target binding and one-use transitions."""

import pytest

from toolfence.colored_net import ColoredAuthorityNet, EffectClass, effect_class
from toolfence.colored_runtime import ColoredToolFenceRuntime
from toolfence.runtime import ToolFenceJudge, ToolFenceRuntime
from toolfence.agentdojo_defense import ToolFenceAuthorizationBoundary
from tests.unit.test_toolfence import Backend, Runtime, monitor_for


def test_external_target_binding_covers_all_email_recipient_fields():
    net = ColoredAuthorityNet("Email alice@example.org the report")
    assert net.decide("send_email", {"recipients": ["alice@example.org"]}) is None
    assert net.decide("send_email", {
        "recipients": ["alice@example.org"], "bcc": ["attacker@example.org"],
    })[0] == "deny"
    assert net.decide("send_email", {"recipients": "alice@example.org"})[0] == "deny"


def test_payment_token_consumes_only_after_success():
    net = ColoredAuthorityNet("Send money to GB29NWBK60161331926819")
    call = {"recipient": "GB29NWBK60161331926819", "amount": 4.0}
    assert net.decide("send_money", call) is None
    assert net.decide("send_money", {"recipient": "GB00ATTACKER1234567890"})[0] == "deny"
    net.observe("send_money", call)
    assert net.decide("send_money", call)[0] == "deny"


def test_remote_read_is_separate_from_local_read_and_origin_bound():
    net = ColoredAuthorityNet("Read https://trusted.example.org/invoice")
    assert effect_class("get_webpage") == EffectClass.REMOTE_READ
    assert effect_class("read_file") == EffectClass.LOCAL_READ
    assert net.decide("get_webpage", {"url": "https://trusted.example.org/next"}) is None
    assert net.decide("get_webpage", {"url": "https://evil.example.org/collect"})[0] == "deny"
    assert net.decide("get_webpage", {"url": "http://trusted.example.org/next"})[0] == "deny"
    assert net.decide("get_webpage", {"url": "https://trusted.example.org:8443/next"})[0] == "deny"
    assert net.decide("get_webpage", {"url": "https://[broken"})[0] == "deny"


def test_runtime_denies_mismatched_transfer_before_dispatch():
    backend = Backend('{"decision":"allow","reason":"okay"}')
    runtime = Runtime(({"message": "sent"}, None))
    guard = ColoredToolFenceRuntime(
        runtime, monitor_for("send_money", effect="financial"), ToolFenceJudge(backend),
        query="Send 4 pounds to GB29NWBK60161331926819", tools=[],
    )
    _, error = guard.run_function(None, "send_money", {
        "recipient": "GB00ATTACKER1234567890", "amount": 4.0,
        "subject": "Refund", "date": "2022-04-01",
    })
    assert "outside the authenticated request" in error
    assert runtime.calls == []
    assert backend.calls == []


def test_runtime_consumes_payment_token_only_after_successful_dispatch():
    runtime = Runtime(({"message": "sent"}, None))
    guard = ColoredToolFenceRuntime(
        runtime, monitor_for("send_money", effect="financial"),
        ToolFenceJudge(Backend('{"decision":"allow","reason":"user requested transfer"}')),
        query="Send 4 pounds to GB29NWBK60161331926819", tools=[],
    )
    args = {"recipient": "GB29NWBK60161331926819", "amount": 4.0,
            "subject": "Refund", "date": "2022-04-01"}
    _, error = guard.run_function(None, "send_money", args)
    assert error is None
    assert len(runtime.calls) == 1
    _, error = guard.run_function(None, "send_money", args)
    assert "already been used" in error
    assert len(runtime.calls) == 1


def test_colored_net_closes_hybrid_judge_target_substitution():
    """An overly permissive judge cannot redirect an explicit payment target."""
    request = "Send 4 pounds to GB29NWBK60161331926819"
    attack = {"recipient": "US133000000121212121212", "amount": 0.01,
              "subject": "Refund", "date": "2022-04-01"}
    baseline_runtime = Runtime(({"message": "sent"}, None))
    baseline = ToolFenceRuntime(
        baseline_runtime, monitor_for("send_money", effect="financial"),
        ToolFenceJudge(Backend('{"decision":"allow","reason":"approved"}')),
        query=request, tools=[],
    )
    _, baseline_error = baseline.run_function(None, "send_money", attack)
    assert baseline_error is None
    assert len(baseline_runtime.calls) == 1

    guarded_runtime = Runtime(({"message": "sent"}, None))
    guarded = ColoredToolFenceRuntime(
        guarded_runtime, monitor_for("send_money", effect="financial"),
        ToolFenceJudge(Backend('{"decision":"allow","reason":"approved"}')),
        query=request, tools=[],
    )
    _, guarded_error = guarded.run_function(None, "send_money", attack)
    assert "outside the authenticated request" in guarded_error
    assert guarded_runtime.calls == []


def test_toolfence_boundary_requires_pinned_authority_and_enables_strict_grants(tmp_path):
    from types import SimpleNamespace
    from toolfence.semantic.schema import ToolManifest, ToolSemantic
    from toolfence.guarded_net import GuardedTransactionNet
    ledger = GuardedTransactionNet(str(tmp_path / "ledger.sqlite"))
    host = {"transaction_net": ledger, "host_bindings": {}, "host_actor": "A",
            "policy_revision": "reviewed-v1",
            "host_reviewer": lambda proposal: True}
    with pytest.raises(ValueError, match="pinned manifest"):
        ToolFenceAuthorizationBoundary(None, None, trusted_catalog=[{"name": "read"}], **host)
    boundary = ToolFenceAuthorizationBoundary(
        SimpleNamespace(pinned_manifest=ToolManifest(tools=[ToolSemantic(
            name="read", effect="read", operation="observe",
        )])), None,
        trusted_catalog=[{"name": "read"}], **host,
    )
    assert boundary.dynamic is True
    assert boundary.grant_cache is not None
    assert boundary.strict_capabilities is True
    assert boundary.host_runtime_kwargs["transaction_net"] is ledger


def test_failed_transfer_keeps_token_for_retry():
    runtime = Runtime(("", "temporary failure"))
    guard = ColoredToolFenceRuntime(
        runtime, monitor_for("send_money", effect="financial"),
        ToolFenceJudge(Backend('{"decision":"allow","reason":"approved"}')),
        query="Send 4 pounds to GB29NWBK60161331926819", tools=[],
    )
    args = {"recipient": "GB29NWBK60161331926819", "amount": 4.0,
            "subject": "Refund", "date": "2022-04-01"}
    _, error = guard.run_function(None, "send_money", args)
    assert error == "temporary failure"
    assert guard.authority_net.snapshot()["consumed_payments"] == []
    runtime.result = ({"message": "sent"}, None)
    _, error = guard.run_function(None, "send_money", args)
    assert error is None
    assert guard.authority_net.snapshot()["consumed_payments"] == [
        "gb29nwbk60161331926819"
    ]
