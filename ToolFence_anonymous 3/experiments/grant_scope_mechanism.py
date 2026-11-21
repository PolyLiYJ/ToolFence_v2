"""Acceptance: one audited grant, five cases, four cumulative scopes (plan 4.1).

No model or network access. Verifies that the *final* decision changes exactly
when the corresponding constraint is added, and that the initial blueprint call
is identical across all four rows.
"""
import argparse
import json
from pathlib import Path

from toolfence.grant_replay import SCOPES, decide, mechanism_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id, snapshot, description in mechanism_cases():
        decisions = {scope.value: decide(snapshot, scope) for scope in SCOPES}
        rows.append({
            "case": case_id,
            "description": description,
            "snapshot": snapshot,
            "decisions": decisions,
        })

    with (args.output / "mechanism.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    header = ["Case", *[s.value for s in SCOPES]]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        cells = [row["case"]] + ["allow" if row["decisions"][s.value] else "deny" for s in SCOPES]
        lines.append("| " + " | ".join(cells) + " |")
    (args.output / "mechanism.md").write_text("\n".join(lines) + "\n")

    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Mechanism acceptance from one audited grant. Each column adds one constraint.}",
           r"\begin{tabular}{l" + "c" * len(SCOPES) + r"}\hline",
           " & ".join(["Case"] + [s.value for s in SCOPES]) + r" \\ \hline"]
    for row in rows:
        cells = [row["case"]] + ["allow" if row["decisions"][s.value] else "deny" for s in SCOPES]
        tex.append(" & ".join(cells) + r" \\")
    (args.output / "mechanism.tex").write_text("\n".join([*tex, r"\hline\end{tabular}\end{table}"]) + "\n")

    expected = {
        "legal": [True] * 4,
        "wrong_source": [True, False, False, False],
        "wrong_stage": [True, True, False, False],
        "budget_exhausted": [True, True, True, False],
        "blueprint": [True] * 4,
    }
    for row in rows:
        actual = [row["decisions"][s.value] for s in SCOPES]
        assert actual == expected[row["case"]], (row["case"], actual)
    print("mechanism acceptance: 5 cases verified, decisions vary by scope", flush=True)


if __name__ == "__main__":
    main()
