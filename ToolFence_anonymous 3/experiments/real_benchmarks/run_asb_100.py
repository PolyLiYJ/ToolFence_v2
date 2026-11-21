from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from .google_backends import build_backends
from .manifest_loader import reviewed_semantics, reviewed_tools_for_context
from .model_profiles import apply_model_profile, normalize_profiles
from .paper_profile import build_paper_profile
from .runner_common import (
    Checkpoint,
    aggregate_rows,
    deterministic_subset,
    ensure_dir,
    sha256_json,
    utc_stamp,
    write_json,
)
from .simple_runtime import SimpleRuntime, execute_calls, request_tool_calls, tool_catalog

BLUEPRINTING = Path("/home/yjli/Agent/Blueprinting")
ASB_ROOT = config.DATA_ROOT / "asb_orig"
if str(BLUEPRINTING) not in sys.path:
    sys.path.insert(0, str(BLUEPRINTING))
from benchmarks.asb_memory import (  # noqa: E402
    ASBMemoryPoisoningBenchmark,
    ASBPlan,
)


def build_client(backends):
    return backends.client


def run_one(backends, profile, benchmark, case, defense, condition):
    started = time.perf_counter()
    attack = condition == "attack"
    visible_tools = [tool.name for tool in case.normal_tools]
    descriptions = {tool.name: tool.description for tool in case.normal_tools}
    context_parts = [f"Agent: {case.agent_name}", f"Task: {case.user_task}"]
    if attack:
        visible_tools.append(case.attack_tool.name)
        descriptions[case.attack_tool.name] = case.attack_tool.description
        context_parts.append(f"Retrieved memory: {case.poisoned_memory}")
    catalog = tool_catalog(visible_tools, descriptions)
    judge_before = 0 if profile is None else backends.judge_snapshot()["request_count"]
    calls = request_tool_calls(
        build_client(backends),
        query=case.user_task,
        visible_context="\n".join(context_parts),
        catalog=catalog,
    )
    runtime = SimpleRuntime(catalog)
    executed, denied = execute_calls(defense, profile, runtime, case.user_task, calls)
    called = tuple(item["tool"] for item in executed)
    evaluation = ASBMemoryPoisoningBenchmark.evaluate(
        case,
        ASBPlan(called_tools=called, response="", memory_found=True if attack else None),
    )
    judge_calls = max(0, int(backends.judge_snapshot()["request_count"] - judge_before))
    return {
        "condition": condition,
        "case_id": f"{case.case_id}:{condition}",
        "agent_name": case.agent_name,
        "utility": bool(evaluation.original_task_success),
        "attack_success": bool(evaluation.attack_success) if attack else False,
        "denied": denied,
        "judge_calls": judge_calls,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "planned_calls": calls,
        "executed_calls": executed,
    }


def build_case_profile(
    backends,
    output_dir: Path,
    judge_name: str,
    defense: str,
    case,
    condition: str,
    catalog: list[dict],
    semantics: dict[str, dict],
    manifest_audit: dict,
):
    """Build a fresh, case-scoped boundary for one ASB case+condition."""
    names = {tool.name for tool in case.normal_tools}
    scoped_catalog = [
        item for item in catalog if item["function"]["name"] in names
    ]
    scoped_semantics = {
        name: semantics[name] for name in names if name in semantics
    }
    ledger = output_dir / "ledgers" / (
        f"asb_{judge_name}_{defense}_{case.case_id.replace(':', '_')}_{condition}.sqlite"
    )
    audit = dict(manifest_audit)
    audit["case_scope"] = sorted(names)
    audit["case_id"] = case.case_id
    audit["condition"] = condition
    return build_paper_profile(
        backends,
        scoped_catalog,
        ledger_path=str(ledger),
        reviewed_semantics=scoped_semantics,
        manifest_audit=audit,
    )


def run_profile(profile_name, output_dir, benchmark, subset, judges, seed):
    model_audit = apply_model_profile(profile_name)
    rows: list[dict] = []
    audits: dict[str, dict] = {}
    for judge_name in judges:
        for defense in ("no_defense", "toolfence_strict"):
            backends = build_backends(judge_name)
            catalog = []
            semantics = {}
            manifest_audit = {}
            if defense == "toolfence_strict":
                catalog, manifest_audit = reviewed_tools_for_context("asb")
                semantics = reviewed_semantics(
                    "asb", [item["function"]["name"] for item in catalog]
                )
                audits[f"{judge_name}:{defense}"] = {
                    "case_isolation": "fresh_boundary_per_case_condition",
                    "tool_scope": "current_case_normal_tools_only",
                    "manifest_audit": manifest_audit,
                }
            checkpoint = Checkpoint(
                output_dir / "checkpoints" / f"asb_{judge_name}_{defense}.jsonl"
            )
            for case in subset:
                for condition in ("clean", "attack"):
                    key = f"{judge_name}:{defense}:{case.case_id}:{condition}"
                    cached = checkpoint.get(key)
                    if cached is None:
                        profile = None
                        if defense == "toolfence_strict":
                            profile = build_case_profile(
                                backends, output_dir, judge_name, defense,
                                case, condition, catalog, semantics, manifest_audit,
                            )
                        row = run_one(backends, profile, benchmark, case, defense, condition)
                        row.update({"judge": judge_name, "defense": defense})
                        checkpoint.add(key, row)
                        cached = checkpoint.get(key)
                    rows.append({k: v for k, v in cached.items() if k != "key"})

    summary = {}
    for judge_name in judges:
        summary[judge_name] = {}
        for defense in ("no_defense", "toolfence_strict"):
            summary[judge_name][defense] = aggregate_rows([
                row for row in rows
                if row.get("judge") == judge_name and row.get("defense") == defense
            ])
    provenance = benchmark.provenance()
    provenance["subset_ids"] = [case.case_id for case in subset]
    provenance["subset_sha256"] = sha256_json([case.to_dict() for case in subset])
    report = {
        "benchmark": "ASB memory poisoning",
        "model_profile": model_audit,
        "subset_size": len(subset),
        "seed": seed,
        "judges": judges,
        "summary": summary,
        "rows": rows,
        "provenance": provenance,
        "profile_audits": audits,
        "config": config.redacted_config(),
        "metric_definition": "Official ASB adapter scoring; strict profile uses reviewed JSON manifest.",
    }
    write_json(output_dir / "asb.json", report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subset-size", type=int, default=config.SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--judges", default="qwen")
    parser.add_argument("--model-profile", default="qwen")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    judges = [item.strip() for item in args.judges.split(",") if item.strip()]
    profiles = normalize_profiles(args.model_profile)
    size = min(6, args.subset_size) if args.smoke else args.subset_size
    output_dir = args.output_dir or (config.RESULTS_ROOT / utc_stamp())
    ensure_dir(output_dir)
    benchmark = ASBMemoryPoisoningBenchmark(source_root=ASB_ROOT, task_limit_per_agent=2)
    subset = deterministic_subset(benchmark.cases, size, args.seed, key=lambda case: case.case_id)
    summary = {}
    for profile_name in profiles:
        profile_dir = output_dir / profile_name
        ensure_dir(profile_dir)
        summary[profile_name] = run_profile(profile_name, profile_dir, benchmark, subset, judges, args.seed)
    print(json.dumps({"benchmark": "ASB", "summary": summary, "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
