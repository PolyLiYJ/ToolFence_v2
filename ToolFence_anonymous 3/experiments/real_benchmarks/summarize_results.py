from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import config

METRIC_FIELDS = [
    "cases_attempted", "cases_completed", "errors", "BU", "U@A", "ASR",
    "denied", "judge_calls", "runtime_seconds",
]


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_summaries(root: Path, manifest_label: str):
    files = [root / "agentdojo.json", root / "asb.json", root / "mcptox.json"]
    files += sorted(root.glob("*/agentdojo.json"))
    files += sorted(root.glob("*/asb.json"))
    files += sorted(root.glob("*/mcptox.json"))
    for path in files:
        data = load(path)
        if not data:
            continue
        benchmark = data.get("benchmark")
        model_profile = data.get("model_profile", {}).get("profile", "qwen")
        for judge, defenses in data.get("summary", {}).items():
            for defense, metrics in defenses.items():
                yield {
                    "benchmark": benchmark,
                    "manifest": manifest_label,
                    "model_profile": model_profile,
                    "judge": judge,
                    "defense": defense,
                    "subset_size": data.get("subset_size"),
                    "seed": data.get("seed"),
                    **metrics,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--old-dir", type=Path, default=config.RESULTS_ROOT / "final_100_qwen")
    parser.add_argument("--old-label", default="heuristic_manifest")
    args = parser.parse_args()

    new_rows = list(iter_summaries(args.result_dir, "reviewed_manifest"))
    old_rows = list(iter_summaries(args.old_dir, args.old_label))
    old_index = {
        (row["benchmark"], row.get("model_profile"), row["judge"], row["defense"]): row
        for row in old_rows
    }

    rows = []
    for row in new_rows:
        key = (row["benchmark"], row.get("model_profile"), row["judge"], row["defense"])
        old = old_index.get(key, {})
        enriched = dict(row)
        for field in METRIC_FIELDS:
            new_value = row.get(field)
            old_value = old.get(field)
            delta = None
            if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                delta = round(float(new_value) - float(old_value), 3)
            enriched[f"old_{field}"] = old_value
            enriched[f"delta_{field}"] = delta
        rows.append(enriched)

    csv_path = args.result_dir / "summary.csv"
    fields = [
        "benchmark", "manifest", "model_profile", "judge", "defense", "subset_size", "seed",
        *METRIC_FIELDS,
        *[f"old_{field}" for field in METRIC_FIELDS],
        *[f"delta_{field}" for field in METRIC_FIELDS],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# ToolFence reviewed manifest 评测结果",
        "",
        f"新结果：`{args.result_dir}`",
        f"旧结果：`{args.old_dir}`",
        "",
        "- 新 manifest 为 JSON reviewed manifest，未配置工具 fail-closed。",
        "- 模型 profile 分为 qwen 与 llama70b；agent/compiler/judge 默认同模型。",
        "- 这些结果仍不是论文主表复现。",
        "",
        "| benchmark | profile | judge | defense | BU | U@A | ASR | old BU | old U@A | old ASR | judge_calls |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row.get('model_profile')} | {row['judge']} | {row['defense']} | "
            f"{row.get('BU')} | {row.get('U@A')} | {row.get('ASR')} | "
            f"{row.get('old_BU')} | {row.get('old_U@A')} | {row.get('old_ASR')} | {row.get('judge_calls')} |"
        )
    (args.result_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
