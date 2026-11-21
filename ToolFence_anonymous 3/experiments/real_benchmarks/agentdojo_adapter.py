from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite

from . import config
from .google_backends import rate_limited_completion
from .manifest_loader import reviewed_semantics, reviewed_tools_for_context
from .paper_profile import PaperProfileBundle, build_paper_profile

BENCHMARK_VERSION = "v1"
ATTACK_STRATEGIES = [
    "direct",
    "ignore_previous",
    "important_instructions",
    "injecagent",
    "system_message",
    "tool_knowledge",
]
NEUTRAL_ATTACK_MODEL_NAME = "the AI language model"

_THOUGHT_SIGNATURES: dict[str, Any] = {}


def _install_gemini_thought_signature_patch() -> None:
    if getattr(openai_llm, "_toolfence_thought_signature_patch", False):
        return
    original_assistant = openai_llm._openai_to_assistant_message
    original_from_tool_call = openai_llm._openai_to_tool_call
    original_tool_call = openai_llm._tool_call_to_openai

    def from_tool_call_with_signature(tool_call: Any) -> Any:
        extra = getattr(tool_call, "extra_content", None)
        if isinstance(extra, Mapping):
            google = extra.get("google")
            if isinstance(google, Mapping):
                signature = google.get("thought_signature")
                if signature and tool_call.id:
                    _THOUGHT_SIGNATURES[tool_call.id] = signature
        return original_from_tool_call(tool_call)

    def assistant_with_signature(message: Any) -> Any:
        signature = None
        extra = getattr(message, "extra_content", None)
        if isinstance(extra, Mapping):
            google = extra.get("google")
            if isinstance(google, Mapping):
                signature = google.get("thought_signature")
        result = original_assistant(message)
        if signature and getattr(result, "tool_calls", None):
            for tool_call in result.tool_calls:
                if tool_call.id:
                    _THOUGHT_SIGNATURES[tool_call.id] = signature
        return result

    def tool_call_with_signature(tool_call: Any) -> Any:
        result = original_tool_call(tool_call)
        signature = _THOUGHT_SIGNATURES.get(tool_call.id)
        if signature and isinstance(result, dict):
            extra = result.setdefault("extra_content", {})
            google = extra.setdefault("google", {})
            google["thought_signature"] = signature
        return result

    openai_llm._openai_to_assistant_message = assistant_with_signature
    openai_llm._openai_to_tool_call = from_tool_call_with_signature
    openai_llm._tool_call_to_openai = tool_call_with_signature
    openai_llm._toolfence_thought_signature_patch = True


_install_gemini_thought_signature_patch()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_trusted_catalog(suite: Any) -> list[dict[str, Any]]:
    catalog = []
    for tool in suite.tools:
        schema = tool.parameters
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            schema = schema.schema()
        catalog.append({
            "type": "function",
            "function": {
                "name": str(tool.name),
                "description": str(tool.description or ""),
                "parameters": dict(schema or {"type": "object", "properties": {}}),
            },
        })
    return catalog


def build_agent_llm(backends: Any) -> OpenAILLM:
    client = backends.client.raw_client
    model = backends.client.model
    def create(**kwargs):
        if "Qwen" in model:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return rate_limited_completion(client, **kwargs)

    gated_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    llm = OpenAILLM(gated_client, model, temperature=0.0)
    llm.name = model
    return llm


def load_attack_with_fallback(pipeline: AgentPipeline, suite: Any, strategy: str) -> Any:
    original_name = pipeline.name
    try:
        pipeline.name = "local"
        attack = load_attack(strategy, suite, pipeline)
    finally:
        pipeline.name = original_name
    if hasattr(attack, "model_name"):
        attack.model_name = NEUTRAL_ATTACK_MODEL_NAME
    return attack


def build_pipeline(
    defense: str,
    suite: Any,
    backends: Any,
    ledger_path: Path,
    suite_name: str | None = None,
) -> tuple[AgentPipeline, PaperProfileBundle | None]:
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=build_agent_llm(backends),
            model_id=None,
            defense=None,
            system_message_name=None,
            system_message=None,
        )
    )
    if defense == "no_defense":
        pipeline.name = f"{pipeline.name}-no_defense"
        return pipeline, None
    if defense != "toolfence_strict":
        raise ValueError(f"Unsupported AgentDojo defense: {defense}")
    catalog, manifest_audit = reviewed_tools_for_context("agentdojo", suite=suite_name)
    semantics = reviewed_semantics(
        "agentdojo", [item["function"]["name"] for item in catalog]
    )
    profile = build_paper_profile(
        backends,
        catalog,
        ledger_path=str(ledger_path),
        reviewed_semantics=semantics,
        manifest_audit=manifest_audit,
    )
    pipeline.elements.insert(2, profile.boundary)
    pipeline.name = f"{pipeline.name}-toolfence_strict"
    return pipeline, profile


def audit_cursor(profile: PaperProfileBundle | None) -> int:
    if profile is None:
        return 0
    return len(profile.boundary.audit_snapshot())


def audit_delta(profile: PaperProfileBundle | None, cursor: int) -> tuple[int, int]:
    if profile is None:
        return 0, 0
    sessions = profile.boundary.audit_snapshot()[cursor:]
    denied = sum(
        1
        for session in sessions
        for decision in session.get("decisions", [])
        if decision.get("state") == "denied"
    )
    return denied, len(sessions)


def run_clean(
    pipeline: AgentPipeline,
    suite: Any,
    user_id: str,
    profile: PaperProfileBundle | None,
    api_before: Mapping[str, int],
) -> dict[str, Any]:
    cursor = audit_cursor(profile)
    started = time.perf_counter()
    row: dict[str, Any] = {
        "condition": "clean",
        "user_task_id": user_id,
        "injection_task_id": None,
        "strategy": None,
    }
    try:
        utility, _ = suite.run_task_with_pipeline(
            pipeline,
            suite.get_user_task_by_id(user_id),
            injection_task=None,
            injections={},
        )
        row["utility"] = bool(utility)
        row["attack_success"] = False
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_seconds"] = round(time.perf_counter() - started, 3)
    denied, _ = audit_delta(profile, cursor)
    row["denied"] = denied
    row["judge_calls"] = max(0, int(profile.backends.judge_snapshot()["request_count"] - api_before["request_count"])) if profile else 0
    return row


def run_attack(
    pipeline: AgentPipeline,
    suite: Any,
    attack: Any,
    user_id: str,
    injection_id: str,
    strategy: str,
    profile: PaperProfileBundle | None,
    api_before: Mapping[str, int],
) -> dict[str, Any]:
    cursor = audit_cursor(profile)
    started = time.perf_counter()
    row: dict[str, Any] = {
        "condition": "attack",
        "user_task_id": user_id,
        "injection_task_id": injection_id,
        "strategy": strategy,
    }
    try:
        user_task = suite.get_user_task_by_id(user_id)
        injection_task = suite.get_injection_task_by_id(injection_id)
        injections = attack.attack(user_task, injection_task)
        utility, security = suite.run_task_with_pipeline(
            pipeline, user_task, injection_task, injections
        )
        row["utility"] = bool(utility)
        row["attack_success"] = bool(security)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_seconds"] = round(time.perf_counter() - started, 3)
    denied, _ = audit_delta(profile, cursor)
    row["denied"] = denied
    row["judge_calls"] = max(0, int(profile.backends.judge_snapshot()["request_count"] - api_before["request_count"])) if profile else 0
    return row
