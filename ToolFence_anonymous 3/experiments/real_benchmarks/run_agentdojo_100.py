from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from . import config
from .agentdojo_adapter import (
    ATTACK_STRATEGIES,
    BENCHMARK_VERSION,
    build_pipeline,
    load_attack_with_fallback,
    load_cases,
    run_attack,
    run_clean,
)
from .google_backends import build_backends
from .model_profiles import apply_model_profile, normalize_profiles
from .runner_common import (
    Checkpoint,
    aggregate_rows,
    deterministic_subset,
    ensure_dir,
    sha256_json,
    utc_stamp,
    write_json,
)

SPLIT_PATH = Path(os.getenv("TOOLFENCE_AGENTDOJO_SPLIT", "data/agentdojo_split/cases.jsonl"))
SOURCE_ROOT = Path(os.getenv("TOOLFENCE_AGENTDOJO_SRC", "data/benchmarks/agentdojo_orig"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_profile(profile_name: str, output_dir: Path, subset, judges, size, seed) -> dict:
    model_audit = apply_model_profile(profile_name)
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for case in subset:
        by_suite[case["suite"]].append(case)

    result_rows: list[dict] = []
    profile_audits: dict[str, dict] = {}
    api_audits: dict[str, dict] = {}
    for judge_name in judges:
        backends = build_backends(judge_name)
        for defense in ("no_defense", "toolfence_strict"):
            for suite_name, cases in by_suite.items():
                label = f"{judge_name}:{defense}:{suite_name}"
                ledger = output_dir / "ledgers" / f"{label.replace(':', '_')}.sqlite"
                suite = get_suite(BENCHMARK_VERSION, suite_name)
                pipeline, profile = build_pipeline(
                    defense, suite, backends, ledger, suite_name=suite_name
                )
                attacks = {
                    strategy: load_attack_with_fallback(pipeline, suite, strategy)
                    for strategy in ATTACK_STRATEGIES
                }
                if profile is not None:
                    profile_audits[label] = profile.audit

                checkpoint = Checkpoint(
                    output_dir / "checkpoints" / f"{label.replace(':', '_')}.jsonl"
                )
                clean_users = sorted({case["user_task_id"] for case in cases})
                for user_id in clean_users:
                    key = f"{label}:clean:{user_id}"
                    cached = checkpoint.get(key)
                    if cached is None:
                        if profile is not None:
                            profile.new_session()
                        row = run_clean(
                            pipeline, suite, user_id, profile,
                            {"request_count": backends.judge_snapshot()["request_count"]},
                        )
                        row.update({
                            "judge": judge_name, "defense": defense,
                            "suite": suite_name, "case_id": user_id,
                        })
                        checkpoint.add(key, row)
                        cached = checkpoint.get(key)
                    result_rows.append({k: v for k, v in cached.items() if k != "key"})

                for case in cases:
                    case_id = f"{case['user_task_id']}__{case['injection_task_id']}__{case['strategy']}"
                    key = f"{label}:attack:{case_id}"
                    cached = checkpoint.get(key)
                    if cached is None:
                        if profile is not None:
                            profile.new_session()
                        row = run_attack(
                            pipeline,
                            suite,
                            attacks[case["strategy"]],
                            case["user_task_id"],
                            case["injection_task_id"],
                            case["strategy"],
                            profile,
                            {"request_count": backends.judge_snapshot()["request_count"]},
                        )
                        row.update({
                            "judge": judge_name, "defense": defense,
                            "suite": suite_name, "case_id": case_id,
                        })
                        checkpoint.add(key, row)
                        cached = checkpoint.get(key)
                    result_rows.append({k: v for k, v in cached.items() if k != "key"})
        api_audits[judge_name] = {
            "compiler_and_agent": backends.snapshot(),
            "judge": backends.judge_snapshot(),
        }

    summary = {}
    for judge_name in judges:
        summary[judge_name] = {}
        for defense in ("no_defense", "toolfence_strict"):
            rows = [
                row for row in result_rows
                if row.get("judge") == judge_name and row.get("defense") == defense
            ]
            summary[judge_name][defense] = aggregate_rows(rows)

    report = {
        "benchmark": "AgentDojo",
        "model_profile": model_audit,
        "subset_size": len(subset),
        "seed": seed,
        "benchmark_version": BENCHMARK_VERSION,
        "attack_strategies": ATTACK_STRATEGIES,
        "judges": judges,
        "summary": summary,
        "rows": result_rows,
        "provenance": {
            "split_path": str(SPLIT_PATH),
            "split_sha256": file_sha256(SPLIT_PATH),
            "source_root": str(SOURCE_ROOT),
            "subset_ids": [
                f"{c['suite']}:{c['user_task_id']}:{c['injection_task_id']}:{c['strategy']}"
                for c in subset
            ],
            "subset_sha256": sha256_json(subset),
        },
        "profile_audits": profile_audits,
        "api_audits": api_audits,
        "config": config.redacted_config(),
        "metric_definition": "Official AgentDojo suite verifier; reviewed JSON manifest strict profile.",
    }
    write_json(output_dir / "agentdojo.json", report)
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
    all_cases = load_cases(SPLIT_PATH)
    subset = deterministic_subset(
        all_cases,
        size,
        args.seed,
        key=lambda row: f"{row['suite']}:{row['user_task_id']}:{row['injection_task_id']}",
    )
    summary = {}
    for index, case in enumerate(subset):
        case["strategy"] = ATTACK_STRATEGIES[index % len(ATTACK_STRATEGIES)]
    for profile_name in profiles:
        profile_dir = output_dir / profile_name if len(profiles) > 1 else output_dir / profile_name
        ensure_dir(profile_dir)
        summary[profile_name] = run_profile(profile_name, profile_dir, subset, judges, size, args.seed)
    print(json.dumps({"benchmark": "AgentDojo", "summary": summary, "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
