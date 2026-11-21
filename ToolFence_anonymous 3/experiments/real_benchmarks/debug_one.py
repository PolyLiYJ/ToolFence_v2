import json
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from experiments.real_benchmarks.agentdojo_adapter import build_pipeline
from experiments.real_benchmarks.google_backends import build_backends

suite = get_suite("v1", "workspace")
backends = build_backends()
pipeline, profile = build_pipeline(
    "toolfence_strict", suite, backends, Path("/tmp/ad_debug3.sqlite")
)
_original_backend = profile.judge.backend


def _logged_backend(system_prompt, user_prompt, max_tokens=256):
    output = _original_backend(system_prompt, user_prompt, max_tokens=max_tokens)
    print("JUDGE_RAW", repr(output[:500]))
    return output


profile.judge.backend = _logged_backend
utility, security = suite.run_task_with_pipeline(
    pipeline, suite.get_user_task_by_id("user_task_14"), None, {}
)
print("RESULT", utility, security)
audit = profile.boundary.audit_snapshot()
print("SESSIONS", len(audit))
for session in audit[-1:]:
    print(json.dumps({
        "plan": session.get("blueprint_plan"),
        "decisions": session.get("decisions"),
        "compiler_error": session.get("compiler_error"),
    }, default=str)[:6000])
