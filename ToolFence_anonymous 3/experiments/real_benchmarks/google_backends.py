from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import config


class _RateGate:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.last + config.MIN_INTERVAL - now
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


API_GATE = _RateGate()


def rate_limited_completion(client: Any, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        API_GATE.wait()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            retryable = status in {429, 500, 502, 503, 504} or status is None
            if not retryable or attempt == config.MAX_RETRIES:
                raise
            time.sleep(min(60.0, 3.0 * (2 ** attempt)))
    raise RuntimeError(f"rate-limited completion failed: {last_error}")


class QwenChatClient:
    """Local Qwen3-8B chat client with the same interface as GoogleChatClient."""

    def __init__(self) -> None:
        import openai

        self.model = config.QWEN_MODEL
        self.raw_client = openai.OpenAI(
            api_key="EMPTY", base_url=config.QWEN_BASE_URL, timeout=config.TIMEOUT, max_retries=0
        )
        self.lock = threading.Lock()
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.0) -> str:
        with self.lock:
            self.request_count += 1
        try:
            response = self.raw_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("empty Qwen response")
            with self.lock:
                self.success_count += 1
                usage = getattr(response, "usage", None)
                if usage:
                    self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            return content
        except Exception:
            with self.lock:
                self.failed_count += 1
            raise

    def audit_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model": self.model,
                "base_url": config.QWEN_BASE_URL,
                "request_count": self.request_count,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }


class GoogleChatClient:
    """OpenAI-compatible Google AI Studio client with bounded retries."""

    def __init__(self) -> None:
        import openai

        self.model = config.GOOGLE_MODEL
        self.raw_client = self.client = openai.OpenAI(
            api_key=config.require_api_key(),
            base_url=config.GOOGLE_BASE_URL,
            timeout=config.TIMEOUT,
            max_retries=0,
        )
        self.lock = threading.Lock()
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.retry_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.error_summary: dict[str, int] = {}

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(config.MAX_RETRIES + 1):
            with self.lock:
                self.request_count += 1
            try:
                API_GATE.wait()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("empty model response")
                usage = getattr(response, "usage", None)
                with self.lock:
                    self.success_count += 1
                    if usage is not None:
                        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                return content
            except Exception as exc:
                last_error = exc
                key = type(exc).__name__
                with self.lock:
                    self.failed_count += 1
                    self.error_summary[key] = self.error_summary.get(key, 0) + 1
                status = getattr(exc, "status_code", None)
                retryable = status in {429, 500, 502, 503, 504} or status is None
                if not retryable or attempt == config.MAX_RETRIES:
                    break
                with self.lock:
                    self.retry_count += 1
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
        raise RuntimeError(f"Google chat failed after retries: {type(last_error).__name__}: {last_error}")

    def audit_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model": self.model,
                "base_url": config.GOOGLE_BASE_URL,
                "request_count": self.request_count,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "retry_count": self.retry_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "error_summary": dict(self.error_summary),
            }


class GoogleCompilerBackend:
    """Backend contract required by SemanticPolicyCompiler."""

    def __init__(self, client: GoogleChatClient):
        self.client = client

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        suffix = "\n\nReturn a single JSON object and no other text."
        return self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + suffix},
            ],
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def audit_snapshot(self) -> dict[str, Any]:
        return self.client.audit_snapshot()


def _normalize_judge_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            decision = payload.get("decision")
            if isinstance(decision, str):
                return json.dumps({
                    "decision": decision.strip().lower(),
                    "reason": str(payload.get("reason", ""))[:200],
                }, ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            pass
    return text


JUDGE_SUFFIX = (
    "\n\nOutput exactly one compact JSON object of the form "
    "{\"decision\":\"allow\"|\"deny\",\"reason\":\"brief reason\"}. "
    "For capability-grant prompts you may use \"grant\" instead of \"allow\". "
    "No markdown, no prose."
)


class GoogleJudgeBackend:
    """Backend contract required by RuntimeJudge."""

    def __init__(self, client: GoogleChatClient):
        self.client = client

    def __call__(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        raw = self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + JUDGE_SUFFIX},
            ],
            max_tokens=max(128, min(max_tokens, config.JUDGE_MAX_TOKENS)),
        )
        return _normalize_judge_text(raw)

    def audit_snapshot(self) -> dict[str, Any]:
        return self.client.audit_snapshot()


class QwenJudgeBackend:
    """Local Qwen3-8B judge served by vLLM on port 8000."""

    def __init__(self) -> None:
        import openai

        self.model = config.QWEN_MODEL
        self.client = openai.OpenAI(
            api_key="EMPTY",
            base_url=config.QWEN_BASE_URL,
            timeout=config.TIMEOUT,
            max_retries=0,
        )
        self.lock = threading.Lock()
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        with self.lock:
            self.request_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + JUDGE_SUFFIX},
                ],
                temperature=0.0,
                max_tokens=max(128, min(max_tokens, config.JUDGE_MAX_TOKENS)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            with self.lock:
                self.success_count += 1
                usage = getattr(response, "usage", None)
                if usage:
                    self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            return _normalize_judge_text(response.choices[0].message.content or "")
        except Exception:
            with self.lock:
                self.failed_count += 1
            raise

    def audit_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model": self.model,
                "base_url": config.QWEN_BASE_URL,
                "request_count": self.request_count,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }


@dataclass
class ModelBackends:
    client: GoogleChatClient
    compiler: GoogleCompilerBackend
    judge: Any
    judge_name: str
    audit: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Mapping[str, Any]:
        return self.client.audit_snapshot()

    def judge_snapshot(self) -> Mapping[str, Any]:
        return self.judge.audit_snapshot()


def build_backends(judge_name: str = "gemini", planner_name: str | None = None) -> ModelBackends:
    planner = planner_name or os.getenv("PLANNER_BACKEND", "gemini")
    if planner == "gemini":
        client: Any = GoogleChatClient()
    elif planner == "qwen":
        client = QwenChatClient()
    else:
        raise ValueError(f"Unsupported planner backend: {planner}")
    if judge_name == "gemini":
        judge: Any = GoogleJudgeBackend(GoogleChatClient())
    elif judge_name == "qwen":
        judge = QwenJudgeBackend()
    else:
        raise ValueError(f"Unsupported judge backend: {judge_name}")
    return ModelBackends(
        client=client,
        compiler=GoogleCompilerBackend(client),
        judge=judge,
        judge_name=judge_name,
    )
