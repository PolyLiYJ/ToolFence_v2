"""Runtime wrapper enforcing per-action authorization decisions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from .judge import JudgeDecision, RuntimeJudge
from .effects import heuristic_effect
from .prompts import denial_message, tool_index
from .workflow import verified_password_otp_step, verified_workflow_prerequisite


class OpenAICompatibleJudgeBackend:
    """Minimal OpenAI-compatible client for the runtime judge."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ):
        import openai

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.observed_model_ids: set[str] = set()
        self.max_output_tokens_used: set[int] = set()
        self.client = openai.OpenAI(
            api_key=api_key, base_url=self.base_url, timeout=timeout, max_retries=0,
        )
        self.temperature = temperature

    def __call__(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 256,
    ) -> str:
        self.max_output_tokens_used.add(int(max_tokens))
        request = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.model.lower().startswith("gpt-5"):
            request["max_completion_tokens"] = max_tokens
        else:
            request["temperature"] = self.temperature
            request["max_tokens"] = max_tokens
            if "siliconflow.cn" in self.base_url and self.model in {
                "Qwen/Qwen3-8B", "Qwen/Qwen3.5-35B-A3B",
            }:
                request["extra_body"] = {"enable_thinking": False}
        self.request_count += 1
        try:
            response = self.client.chat.completions.create(**request)
        except Exception:
            self.failed_count += 1
            raise
        self.success_count += 1
        if getattr(response, "model", None):
            self.observed_model_ids.add(str(response.model))
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Runtime judge returned empty content")
        return content

    def audit_snapshot(self) -> dict[str, object]:
        return {
            "requested_model_id": self.model,
            "observed_model_ids": sorted(self.observed_model_ids),
            "api_call_count": self.request_count,
            "successful_api_calls": self.success_count,
            "failed_api_calls": self.failed_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "temperature": self.temperature if not self.model.lower().startswith("gpt-5") else None,
            "max_output_tokens": sorted(self.max_output_tokens_used),
            "endpoint": self.base_url,
        }


def _scalars(value: Any):
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _scalars(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _scalars(item)
        return
    yield str(value)


def compute_provenance(
    query: str,
    observed_text: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Label each argument and capture the excerpt supporting that label.

    Shared by the pure runtime guard and the hybrid (v2 + judge) runtime so
    both routes see identical provenance evidence.
    """
    labels: Dict[str, str] = {}
    evidence: Dict[str, Dict[str, str]] = {}
    query_norm = (query or "").casefold()
    for name, value in arguments.items():
        scalars = [item for item in _scalars(value) if item]
        source = "model"
        excerpt = ""
        for item in scalars:
            # A one-digit user count (e.g. "3 results") is still attested;
            # match it as a whole token so 3 does not match 30 or 13.
            in_request = (
                bool(re.search(r"(?<!\w)" + re.escape(item.casefold()) + r"(?!\w)", query_norm))
                if item.isdigit() else len(item) >= 2 and item.casefold() in query_norm
            )
            if in_request:
                source = "user"
                excerpt = item[:200]
                break
        if source == "model":
            for tool_name, text in (observed_text or {}).items():
                position = next(
                    (
                        text.find(item) for item in scalars
                        if len(item) >= 2 and item in text
                    ),
                    -1,
                )
                if position >= 0:
                    source = f"tool:{tool_name}"
                    start = max(0, position - 120)
                    excerpt = text[start:start + 400]
                    break
        labels[name] = source
        if excerpt:
            evidence[name] = {"provenance": source, "excerpt": excerpt}
    return labels, evidence


class RuntimeGuardRuntime:
    """Wrap a FunctionsRuntime and authorize every call before dispatch.

    Unlike the blueprint design, tools are never hidden from the agent: an
    unauthorized call is denied with an explicit, recoverable message instead of
    silently disappearing from the tool list.
    """

    def __init__(
        self,
        runtime: Any,
        judge: RuntimeJudge,
        *,
        query: str,
        tools: List[Dict[str, Any]],
        effects: Optional[Dict[str, str]] = None,
        workflow_context: bool = False,
    ):
        self.runtime = runtime
        self.judge = judge
        self.query = query
        self.tools = tool_index(tools)
        # Optional controller-owned effect classification. When present,
        # read/local-compute calls are authorized without a judge call: they
        # cannot complete an injection goal, and judging them only starves the
        # agent. Unknown effects still go to the judge.
        self.effects = effects or {}
        self.workflow_context = bool(workflow_context)
        self.completed_tools: List[str] = []
        self.structured_observations: Dict[str, Any] = {}
        self.available_read_tools = sorted(
            name for name, info in self.tools.items()
            if heuristic_effect(name, info.get("description", "")) == "read"
        )
        self.read_effects = {"read", "local_compute"}
        self.decisions: List[Dict[str, Any]] = []
        self.observed: Dict[str, set] = {}
        self.observed_text: Dict[str, str] = {}

    @property
    def functions(self) -> Any:
        return self.runtime.functions

    def register_function(self, function: Any) -> Any:
        return self.runtime.register_function(function)

    def _provenance(
        self, arguments: Mapping[str, Any],
    ) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        """Label each argument and capture the excerpt supporting that label."""
        labels: Dict[str, str] = {}
        evidence: Dict[str, Dict[str, str]] = {}
        query_norm = self.query.casefold()
        for name, value in arguments.items():
            scalars = [item for item in _scalars(value) if item]
            source = "model"
            excerpt = ""
            for item in scalars:
                if len(item) >= 2 and item.casefold() in query_norm:
                    source = "user"
                    excerpt = item[:200]
                    break
            if source == "model":
                for tool_name, text in self.observed_text.items():
                    position = next(
                        (
                            text.find(item) for item in scalars
                            if len(item) >= 2 and item in text
                        ),
                        -1,
                    )
                    if position >= 0:
                        source = f"tool:{tool_name}"
                        start = max(0, position - 120)
                        excerpt = text[start:start + 400]
                        break
            labels[name] = source
            if excerpt:
                evidence[name] = {"provenance": source, "excerpt": excerpt}
        return labels, evidence

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Any:
        normalized = {
            name: value for name, value in dict(kwargs).items() if value is not None
        }
        tool = self.tools.get(function, {"name": function})
        provenance, evidence = self._provenance(normalized)
        effect = self.effects.get(function)
        # Read routing is provenance-conditional. A read whose arguments all
        # occur verbatim in the authenticated request (or which takes no
        # arguments) cannot introduce a new access target, so it may bypass the
        # judge. A read whose arguments came from tool output or from the model
        # can: AgentDojo's slack injection task is completed purely by fetching
        # an attacker URL, so unconditionally trusting reads measured 60% ASR on
        # slack. Those calls keep the judge.
        read_trusted = effect in self.read_effects and all(
            source == "user" for source in provenance.values()
        )
        workflow_trusted = self.workflow_context and (
            verified_workflow_prerequisite(
                self.query, function, normalized, self.structured_observations,
            )
            or verified_password_otp_step(
                self.query, function, normalized, env, self.decisions,
                self.completed_tools,
            )
        )
        if read_trusted:
            decision = JudgeDecision(allowed=True, reason="read with authenticated arguments")
            routed = "read_auto_allow"
        elif workflow_trusted:
            decision = JudgeDecision(allowed=True, reason="verified shopping workflow dependency")
            routed = "workflow_dependency_allow"
        else:
            decision = self.judge.decide(
                query=self.query, tool=tool, arguments=normalized,
                provenance=provenance, evidence=evidence,
                task_state=(
                    {
                        "observed_tool_calls": self.completed_tools[-12:],
                        "available_read_tools": self.available_read_tools,
                    }
                    if self.workflow_context else None
                ),
            )
            routed = "judge"
        self.decisions.append({
            "tool": function,
            "arguments": dict(normalized),
            "provenance": provenance,
            "evidence": {name: info["provenance"] for name, info in evidence.items()},
            "route": routed,
            "effect": effect,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "parsed": decision.parsed,
            "error": decision.error,
            "status": decision.status or ("allow" if decision.allowed else "deny"),
            "next_step": decision.next_step,
        })
        if not decision.allowed:
            message = denial_message(
                decision.reason,
                denied_tool=function,
                authenticated_request=self.query,
                next_step=(
                    decision.next_step
                    if self.workflow_context and decision.next_step in self.available_read_tools
                    else ""
                ),
            )
            if raise_on_error:
                raise PermissionError(message)
            return "", message
        value, error = self.runtime.run_function(
            env, function, normalized, raise_on_error=raise_on_error,
        )
        if error is None:
            self.completed_tools.append(function)
            if function in {
                "get_shopping_account_user_information", "view_cart",
                "view_available_discounts",
            }:
                self.structured_observations[function] = value
            if function == "search_product" and isinstance(value, Mapping):
                self.structured_observations.setdefault(function, []).append(value)
            try:
                rendered = value if isinstance(value, str) else json.dumps(value, default=str)
            except TypeError:
                rendered = str(value)
            self.observed.setdefault(function, set()).update(
                item for item in _scalars(rendered) if len(item) >= 2
            )
            # Keep the raw observation so argument provenance can show the judge
            # the exact context a value was copied from.
            previous = self.observed_text.get(function, "")
            merged = (previous + "\n" + rendered) if previous else rendered
            self.observed_text[function] = merged[:4000]
        return value, error
