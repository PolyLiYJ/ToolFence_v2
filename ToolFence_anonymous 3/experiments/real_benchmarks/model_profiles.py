from __future__ import annotations

import os
from typing import Any

from . import config

PROFILES: dict[str, dict[str, Any]] = {
    "qwen": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen/Qwen3-8B",
        "planner_backend": "qwen",
        "tool_call_parser": "hermes",
        "max_model_len": 8192,
    },
    "llama70b": {
        "base_url": "http://127.0.0.1:8002/v1",
        "model": "/media/ssd3/models/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "planner_backend": "qwen",
        "tool_call_parser": "llama3_json",
        "max_model_len": 4096,
    },
}


def normalize_profiles(value: str) -> list[str]:
    if value == "both":
        return ["qwen", "llama70b"]
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("model profile is empty")
    for name in names:
        if name not in PROFILES:
            raise ValueError(f"Unknown model profile: {name}")
    return names


def apply_model_profile(name: str) -> dict[str, Any]:
    if name not in PROFILES:
        raise ValueError(f"Unknown model profile: {name}")
    profile = PROFILES[name]
    config.QWEN_BASE_URL = profile["base_url"]
    config.QWEN_MODEL = profile["model"]
    os.environ["PLANNER_BACKEND"] = profile["planner_backend"]
    return {
        "profile": name,
        "base_url": profile["base_url"],
        "model": profile["model"],
        "planner_backend": profile["planner_backend"],
        "tool_call_parser": profile["tool_call_parser"],
        "max_model_len": profile["max_model_len"],
    }
