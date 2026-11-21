import json
from types import SimpleNamespace

import pytest

from toolfence.semantic.architect import OpenAICompatibleJSONBackend, SemanticPolicyCompiler


def test_llama_cpp_sampler_error_retries_without_transport_json_mode():
    backend = OpenAICompatibleJSONBackend(
        model="gemma3:4b", base_url="http://127.0.0.1:8010/v1", api_key="local",
    )
    requests = []

    class SamplerError(Exception):
        status_code = 400

    def create(**request):
        requests.append(request)
        if len(requests) == 1:
            raise SamplerError("Failed to initialize samplers: std::exception")
        return SimpleNamespace(
            model="gemma3:4b", usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        )

    backend.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    assert backend("JSON only", "Return ok") == '{"ok":true}'
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in requests[1]
    assert backend.audit_snapshot()["json_mode_fallbacks"] == 1
from toolfence.semantic.monitor import SemanticMonitor
from toolfence.semantic.prompts import (
    PLAN_SYSTEM_PROMPT,
    authorization_plan_prompt,
)
from toolfence.semantic.runtime import ToolFenceSemanticRuntime
from toolfence.semantic.schema import AuthorizationPlan, ToolManifest


SEARCH_EMAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_emails",
        "description": "Search received emails by query and optional sender.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "sender": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


def test_plan_prompt_includes_controller_owned_tool_descriptions():
    manifest = {
        "tools": [{
            "name": "search_emails",
            "effect": "read",
            "operation": "observe",
            "resource": "inbox_emails",
            "reversible": True,
            "parameters": {},
        }],
    }
    prompt = json.loads(authorization_plan_prompt(
        "Find the security code", manifest, [SEARCH_EMAILS_TOOL],
    ))
    assert prompt["tool_contracts"][0]["description"] == (
        "Search received emails by query and optional sender."
    )
    assert "every step" in PLAN_SYSTEM_PROMPT
    assert "exact-filename lookup" in PLAN_SYSTEM_PROMPT
    assert "multi-tool workflow" in PLAN_SYSTEM_PROMPT
    assert "read_channel_messages" in PLAN_SYSTEM_PROMPT
    assert "get_webpage" in PLAN_SYSTEM_PROMPT
    assert "send_direct_message" in PLAN_SYSTEM_PROMPT
    assert "Refund GB00EXAMPLE" in PLAN_SYSTEM_PROMPT


def test_optional_null_argument_is_normalized_to_omission():
    manifest = ToolManifest.model_validate({
        "tools": [{
            "name": "search_emails",
            "effect": "read",
            "operation": "observe",
            "resource": "inbox_emails",
            "reversible": True,
            "parameters": {
                "query": {
                    "semantic_role": "search_query",
                    "authority_sensitive": False,
                },
                "sender": {
                    "semantic_role": "search_sender_scope",
                    "authority_sensitive": False,
                },
            },
        }],
    })
    plan = AuthorizationPlan.model_validate({
        "goal": "Find the Facebook security code",
        "capabilities": [{
            "tool": "search_emails",
            "max_calls": 1,
            "parameters": {
                "query": {"binding": "literal", "value": "Facebook security code"},
            },
        }],
    })
    compiler = SemanticPolicyCompiler(lambda _system, _user: {})
    normalized_tools = [compiler._normalise_tool(SEARCH_EMAILS_TOOL)]
    validated = compiler._validate_plan(
        "Find my Facebook security code", plan, manifest, normalized_tools,
    )
    blueprint_tools = {
        "search_emails": compiler._parameter_rules(normalized_tools[0]),
    }

    from toolfence.foundation.blueprint import SessionBlueprint, ToolBlueprint, ToolPermission

    blueprint = SessionBlueprint(
        session_id="optional-null",
        user_intent="Find my Facebook security code",
        authorized_goals=[validated.goal],
        allowed_tools={
            "search_emails": ToolBlueprint(
                tool_id="search_emails",
                permission=ToolPermission.ALLOWED,
                parameter_rules=blueprint_tools["search_emails"],
                max_calls_per_session=8,
            ),
        },
        max_total_tool_calls=8,
    )

    class Runtime:
        functions = {"search_emails": object()}

        def __init__(self):
            self.calls = []

        def run_function(self, env, function, kwargs, raise_on_error=False):
            self.calls.append((function, kwargs))
            return [], None

    raw = Runtime()
    monitor = SemanticMonitor(blueprint, validated, manifest)
    monitor.register_user_query("Find my Facebook security code")
    runtime = ToolFenceSemanticRuntime(raw, monitor)
    _, error = runtime.run_function(None, "search_emails", {
        "query": "Facebook security code",
        "sender": None,
    })
    assert error is None
    assert raw.calls == [("search_emails", {"query": "Facebook security code"})]


def test_generated_temporal_filter_is_rejected():
    calendar_tool = {
        "name": "search_calendar_events",
        "description": "Search calendar events, optionally filtering by date.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["query"],
        },
    }
    manifest = ToolManifest.model_validate({
        "tools": [{
            "name": "search_calendar_events",
            "effect": "read",
            "operation": "observe",
            "resource": "calendar_events",
            "reversible": True,
            "parameters": {
                "query": {"semantic_role": "search_query", "authority_sensitive": False},
                "date": {"semantic_role": "date_filter", "authority_sensitive": False},
            },
        }],
    })
    plan = AuthorizationPlan.model_validate({
        "goal": "Find the dinner",
        "capabilities": [{
            "tool": "search_calendar_events",
            "parameters": {
                "query": {"binding": "literal", "value": "Dinner"},
                "date": {"binding": "generated"},
            },
        }],
    })
    with pytest.raises(ValueError, match="Temporal parameter cannot be generated"):
        SemanticPolicyCompiler._validate_plan(
            "Where is the Dinner on May 24th?", plan, manifest, [calendar_tool],
        )


def test_local_schema_ref_preserves_enum_values():
    raw = {
        "type": "object",
        "$defs": {
            "Permission": {"type": "string", "enum": ["r", "rw"]},
        },
        "properties": {
            "permission": {"$ref": "#/$defs/Permission"},
        },
        "required": ["permission"],
    }
    compact = SemanticPolicyCompiler._compact_object_schema(raw)
    assert compact["properties"]["permission"]["enum"] == ["r", "rw"]


def test_plural_derived_side_effect_gets_bounded_multi_call_budget():
    tools = [
        {
            "name": "search_files",
            "description": "Search files by content.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email to recipients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipients": {"type": "array"},
                    "body": {"type": "string"},
                },
                "required": ["recipients", "body"],
            },
        },
    ]
    manifest = ToolManifest.model_validate({
        "tools": [
            {
                "name": "search_files", "effect": "read", "operation": "observe",
                "resource": "cloud_files", "reversible": True,
                "parameters": {
                    "query": {"semantic_role": "search_query", "authority_sensitive": False},
                },
            },
            {
                "name": "send_email", "effect": "communication",
                "operation": "communicate", "resource": "email", "reversible": False,
                "parameters": {
                    "recipients": {
                        "semantic_role": "message_recipient",
                        "authority_sensitive": True,
                    },
                    "body": {"semantic_role": "message_body", "authority_sensitive": False},
                },
            },
        ],
    })
    plan = AuthorizationPlan.model_validate({
        "goal": "Email each person in the TODO list",
        "capabilities": [
            {
                "tool": "search_files", "max_calls": 1,
                "parameters": {"query": {"binding": "literal", "value": "TODO list"}},
            },
            {
                "tool": "send_email", "evidence": "send an email", "max_calls": 20,
                "parameters": {
                    "recipients": {
                        "binding": "derived", "source_tools": ["search_files"],
                    },
                    "body": {
                        "binding": "computed", "source_tools": ["search_files"],
                    },
                },
            },
        ],
    })
    validated = SemanticPolicyCompiler._validate_plan(
        "Please send an email to each person in the TODO list.",
        plan,
        manifest,
        tools,
        [],
        8,
    )
    assert validated.by_tool()["send_email"].max_calls == 8


def test_share_side_effect_is_grounded_by_explicit_share_request():
    share_tool = {
        "name": "share_file",
        "description": "Share a file with a user.",
        "parameters": {
            "type": "object",
            "$defs": {
                "SharingPermission": {"type": "string", "enum": ["r", "rw"]},
            },
            "properties": {
                "file_id": {"type": "string"},
                "email": {"type": "string"},
                "permission": {"$ref": "#/$defs/SharingPermission"},
            },
            "required": ["file_id", "email", "permission"],
        },
    }
    manifest = ToolManifest.model_validate({
        "tools": [{
            "name": "share_file",
            "effect": "credential_change",
            "operation": "credential_change",
            "resource": "cloud_file_acl",
            "reversible": True,
            "parameters": {
                "file_id": {"semantic_role": "file_target", "authority_sensitive": True},
                "email": {"semantic_role": "grantee_identity", "authority_sensitive": True},
                "permission": {"semantic_role": "access_level", "authority_sensitive": True},
            },
        }],
    })
    plan = AuthorizationPlan.model_validate({
        "goal": "Share the document",
        "capabilities": [{
            "tool": "share_file",
            "parameters": {
                "file_id": {"binding": "literal", "value": "hawaii-packing-list.docx"},
                "email": {"binding": "literal", "value": "john.doe@gmail.com"},
                "permission": {"binding": "literal", "value": "read"},
            },
        }],
    })
    compiler = SemanticPolicyCompiler(lambda _system, _user: {})
    normalized_share_tool = compiler._normalise_tool({"function": share_tool})
    validated = compiler._validate_plan(
        "Share the document hawaii-packing-list.docx with john.doe@gmail.com "
        "with read permissions.",
        plan,
        manifest,
        [normalized_share_tool],
    )
    assert validated.by_tool()["share_file"].evidence.lower().startswith("share")
    permission = validated.by_tool()["share_file"].parameters["permission"]
    assert permission.binding == "normalized_literal"
    assert permission.value == "r"
    assert permission.inferred_components == ["enum_alias"]


def test_ungrounded_filename_read_is_refined_to_content_search():
    tools = [
        {
            "name": "search_files_by_filename",
            "description": "Search files by exact filename.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
        {
            "name": "search_files",
            "description": "Search files by content.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]
    manifest = ToolManifest.model_validate({
        "tools": [
            {
                "name": "search_files_by_filename", "effect": "read",
                "operation": "observe", "resource": "cloud_file", "reversible": True,
                "parameters": {
                    "filename": {
                        "semantic_role": "file_name_query", "authority_sensitive": False,
                    },
                },
            },
            {
                "name": "search_files", "effect": "read", "operation": "observe",
                "resource": "cloud_file", "reversible": True,
                "parameters": {
                    "query": {
                        "semantic_role": "content_search_query", "authority_sensitive": False,
                    },
                },
            },
        ],
    })
    plan = AuthorizationPlan.model_validate({
        "goal": "Find the document about vacation plans",
        "capabilities": [{
            "tool": "search_files_by_filename",
            "parameters": {"filename": {"binding": "generated"}},
        }],
    })
    contractions = []
    validated = SemanticPolicyCompiler._validate_plan(
        "Find the document about vacation plans", plan, manifest, tools, contractions,
    )
    assert [item.tool for item in validated.capabilities] == ["search_files"]
    assert validated.capabilities[0].parameters["query"].binding == "free"
    assert "ungrounded_filename_read_refined:search_files_by_filename->search_files" in contractions
