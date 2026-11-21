"""Merge replay and clean-run artifacts into the paper's four-row table.

Each row is cumulative: tool -> + source binding -> + workflow stage -> + use
budget. FAR comes from the paired replay of independently labeled unauthorized
proposals; BU and grants/task come from clean tasks. ASR and U@A require
interaction traces under attack and are reported as N/A for the fixed-proposal
protocol. Every value is traceable to the per-sample JSONL and its denominator.
"""
import argparse
import csv
import json
from pathlib import Path

from toolfence.grant_replay import SCOPES

LABELS = {
    "tool": "Tool only",
    "scope": "Tool only",
    "source": "+ source binding",
    "stage": "+ workflow stage",
    "budget": "+ use budget",
}


def load_replay(path):
    """Return {benchmark: {scope: (numerator, denominator)}} from replay.csv."""
    result = {}
    if not path.exists():
        return result
    with path.open() as f:
        for row in csv.DictReader(f):
            n = row.get("Unauthorized", "0")
            false_allow = row.get("False allows", "N/A")
            if not n.isdigit() or false_allow == "N/A" or int(n) == 0:
                value = ("N/A", 0)
            else:
                value = (int(false_allow), int(n))
            result.setdefault(row["Benchmark"], {})[row["Scope"]] = value
    return result


def load_clean(path):
    """Return {benchmark: {scope: (bu_numerator, bu_denominator, grants, n_tasks)}}.

    BU is only defined for benchmarks with a native scorer (AgentDojo); ASB
    keeps a coverage proxy and reports BU as N/A.
    """
    agg = {}
    if not path.exists():
        return agg
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["benchmark"], row["scope"])
            slot = agg.setdefault(key, dict(bu_num=0, bu_den=0, grants=0, tasks=0))
            slot["tasks"] += 1
            slot["grants"] += int(row.get("grant_count", 0) or 0)
            bu = row.get("bu", "")
            if bu in ("True", "False"):
                slot["bu_den"] += 1
                slot["bu_num"] += int(bu == "True")
    return {(b, s): (v["bu_num"], v["bu_den"], v["grants"], v["tasks"])
            for (b, s), v in agg.items()}


def fmt_fraction(value):
    numerator, denominator = value
    if denominator == 0:
        return "N/A"
    return f"{100.0 * numerator / denominator:.2f} ({numerator}/{denominator})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Directory with replay.csv")
    parser.add_argument("--clean", type=Path, required=True, help="Directory with clean_summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    replay = load_replay(args.replay / "replay.csv")
    clean = load_clean(args.clean / "clean_summary.csv")
    benchmarks = ["agentdojo", "asb"]

    header = ["Runtime grant scope", "ASR (%)", "U@A (%)", "BU (%)", "FAR (%)", "Grants/task"]
    rows = []
    for benchmark in benchmarks:
        for s in SCOPES:
            far = replay.get(benchmark, {}).get(s.value, ("N/A", 0))
            b = clean.get((benchmark, s.value))
            bu = f"{100.0*b[0]/b[1]:.2f} ({b[0]}/{b[1]})" if b and b[1] else "N/A"
            grants = f"{b[2]/b[3]:.2f} ({b[2]}/{b[3]})" if b and b[3] else "N/A"
            rows.append([f"{benchmark}: " + LABELS[s.value], "N/A", "N/A", bu,
                         fmt_fraction(far), grants])

    with (args.output / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header, *rows])

    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (args.output / "summary.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Runtime grant scope ablation. Each row adds one constraint. FAR is over independently labeled unauthorized proposals; BU and grants/task are from clean tasks. ASR and U@A require interaction traces.}",
           r"\begin{tabular}{lccccc}\hline",
           " & ".join(header) + r" \\ \hline"]
    tex += [" & ".join(map(str, r)) + r" \\" for r in rows]
    (args.output / "summary.tex").write_text("\n".join([*tex, r"\hline\end{tabular}\end{table}"]) + "\n")

    manifest = dict(
        replay=dict(proposals_jsonl=str((args.replay / "proposals.jsonl").resolve()),
                    coverage_csv=str((args.replay / "coverage.csv").resolve()),
                    replay_csv=str((args.replay / "replay.csv").resolve())),
        clean=dict(summary_csv=str((args.clean / "clean_summary.csv").resolve())),
        note="ASR/U@A are N/A under fixed-proposal replay; a value of N/A means the denominator was zero or the run was not completed.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("report written to", args.output, flush=True)


if __name__ == "__main__":
    main()
