import json
from pathlib import Path

from experiments.real_benchmarks.google_backends import build_backends
from experiments.real_benchmarks.manifest_loader import reviewed_semantics, reviewed_tools_for_context
from experiments.real_benchmarks.model_profiles import apply_model_profile
from experiments.real_benchmarks.paper_profile import build_paper_profile
from experiments.real_benchmarks.run_mcptox_100 import load_cases
from experiments.real_benchmarks.simple_runtime import SimpleRuntime

apply_model_profile("qwen")
backends = build_backends("qwen")
case = [c for c in load_cases() if c["server"] == "Prisma"][0]
catalog, audit = reviewed_tools_for_context("mcptox", server=case["server"])
sem = reviewed_semantics("mcptox", [item["function"]["name"] for item in catalog])
profile = build_paper_profile(
    backends, catalog, ledger_path="/tmp/tf_mcp_debug.sqlite",
    reviewed_semantics=sem, manifest_audit=audit,
)
calls = [{"tool": "migrate-dev", "arguments": {"name": "add-post-title"}}]
runtime = SimpleRuntime(catalog)
_, guarded, _, _, _ = profile.boundary.query(case["query"], runtime)
for call in calls:
    result = guarded.run_function(None, call["tool"], call["arguments"])
    print("RESULT", repr(result)[:600])
print("CALLS", calls)
print("EXEC", runtime.executed)
sessions = profile.boundary.audit_snapshot()
print("SESSIONS", len(sessions))
for session in sessions[-1:]:
    print(json.dumps({
        "plan": session.get("blueprint_plan"),
        "decisions": session.get("decisions"),
        "compiler_error": session.get("compiler_error"),
        "runtime_errors": session.get("runtime_errors"),
    }, default=str)[:8000])
