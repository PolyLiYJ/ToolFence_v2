import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/yjli/Agent/Blueprinting")
from benchmarks.asb_memory import ASBMemoryPoisoningBenchmark

from experiments.real_benchmarks.google_backends import build_backends
from experiments.real_benchmarks.manifest_loader import reviewed_semantics, reviewed_tools_for_context
from experiments.real_benchmarks.model_profiles import apply_model_profile
from experiments.real_benchmarks.paper_profile import build_paper_profile
from experiments.real_benchmarks.simple_runtime import SimpleRuntime

apply_model_profile("qwen")
backends = build_backends("qwen")
catalog, audit = reviewed_tools_for_context("asb")
sem = reviewed_semantics("asb", [item["function"]["name"] for item in catalog])
profile = build_paper_profile(
    backends, catalog, ledger_path="/tmp/tf_manifest_debug.sqlite",
    reviewed_semantics=sem, manifest_audit=audit,
)
bench = ASBMemoryPoisoningBenchmark(source_root=Path("/home/yjli/ToolFence_anonymous3/data/asb_orig"), task_limit_per_agent=2)
case = bench.cases[0]
calls = [{"tool": t.name, "arguments": {}} for t in case.normal_tools]
runtime = SimpleRuntime(catalog)
_, guarded, _, _, _ = profile.boundary.query(case.user_task, runtime)
for call in calls:
    result = guarded.run_function(None, call["tool"], call["arguments"])
    print("RESULT", call["tool"], repr(result)[:600])
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
