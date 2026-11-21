from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_real"
DATA_ROOT = ROOT / "data"

GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
GOOGLE_BASE_URL = os.getenv(
    "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "models/gemini-3.5-flash-lite")
JUDGE_BACKENDS = tuple(
    item.strip() for item in os.getenv("JUDGE_BACKENDS", "gemini,qwen").split(",") if item.strip()
)
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")

SEED = int(os.getenv("BENCHMARK_SEED", "17"))
SUBSET_SIZE = int(os.getenv("BENCHMARK_SUBSET_SIZE", "100"))
CONCURRENCY = int(os.getenv("BENCHMARK_CONCURRENCY", "4"))
TIMEOUT = float(os.getenv("BENCHMARK_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("BENCHMARK_MAX_RETRIES", "5"))
MIN_INTERVAL = float(os.getenv("BENCHMARK_MIN_INTERVAL", "6.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("BENCHMARK_MAX_OUTPUT_TOKENS", "1024"))
JUDGE_MAX_TOKENS = int(os.getenv("BENCHMARK_JUDGE_MAX_TOKENS", "512"))

POLICY_REVISION = os.getenv("TOOLFENCE_POLICY_REVISION", "paper-profile-gemini-v1")
HOST_ACTOR = os.getenv("TOOLFENCE_HOST_ACTOR", "controller")


def require_api_key() -> str:
    key = os.getenv(GOOGLE_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Missing {GOOGLE_API_KEY_ENV} environment variable")
    return key


def redacted_config() -> dict:
    return {
        "base_url": GOOGLE_BASE_URL,
        "model": GOOGLE_MODEL,
        "judge_backends": list(JUDGE_BACKENDS),
        "qwen_base_url": QWEN_BASE_URL,
        "qwen_model": QWEN_MODEL,
        "api_key_env": GOOGLE_API_KEY_ENV,
        "api_key_value": "<redacted>",
        "seed": SEED,
        "subset_size": SUBSET_SIZE,
        "concurrency": CONCURRENCY,
        "timeout_seconds": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "min_request_interval_seconds": MIN_INTERVAL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "policy_revision": POLICY_REVISION,
        "host_actor": HOST_ACTOR,
        "https_proxy_env": bool(os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")),
    }
