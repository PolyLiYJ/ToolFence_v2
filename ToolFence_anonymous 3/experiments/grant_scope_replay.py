"""Audit historical JSONL runs or replay host-normalized labeled proposals."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from toolfence.grant_replay import Scope, Incomplete, decide, name, require


def audit(path, benchmark, scopes, output, counts):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for line, raw in enumerate(path.open(), 1):
        if not raw.strip():
            continue
        r = json.loads(raw)
        # Do not infer missing host state from an authorization verdict or task label.
        if r.get("format") == "toolfence.replay.v1":
            proposals = [r]
        elif benchmark == "agentdojo":
            proposals = r.get("decisions", [])
        else:
            proposals = [step.get("proposed", {}) for step in r.get("trace", [])]
        for index, event in enumerate(proposals):
            info = dict(benchmark=benchmark, source=str(path), sha256=digest,
                        line=line, index=index, model=r.get("model"), decisions={})
            try:
                if not isinstance(event, dict):
                    raise Incomplete("proposal object")
                if r.get("format") != "toolfence.replay.v1":
                    raise Incomplete("missing verified grant, pre-call state and independent call label")
                if name(r, "benchmark") != benchmark:
                    raise ValueError("benchmark mismatch")
                info["model"] = name(r, "model")
                name(r, "label_source")
                name(r, "query")
                snapshot = require(r, "snapshot", dict)
                if snapshot.get("origin") != "approved_runtime_grant":
                    raise Incomplete("outside runtime grant population")
                unauthorized = require(r, "unauthorized", bool)
                decisions = {s.value: decide(snapshot, s) for s in scopes}
                info.update(decisions=decisions, unauthorized=unauthorized)
                for s, allowed in decisions.items():
                    counts[benchmark,s].update(paired=1, unauthorized=int(unauthorized),
                                                false_allow=int(allowed and unauthorized))
            except Incomplete as exc:
                info["excluded"] = str(exc)
            for s in scopes:
                counts[benchmark,s.value]["candidates"] += 1
            output.write(json.dumps(info) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for bench in ("agentdojo", "asb"):
        parser.add_argument("--" + bench, type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grant-scope", choices=["all", *[s.value for s in Scope]], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scopes = list(Scope) if args.grant_scope == "all" else [Scope(args.grant_scope)]
    counts = {(b,s.value): Counter() for b in ("agentdojo","asb") for s in scopes}
    with (args.output / "proposals.jsonl").open("w") as out:
        for b in ("agentdojo", "asb"):
            seen = set()
            for path in getattr(args,b):
                digest = hashlib.sha256(path.read_bytes()).digest()
                if digest not in seen:
                    audit(path,b,scopes,out,counts)
                seen.add(digest)
    header = ["Benchmark","Scope","Candidates","Paired","Unauthorized","False allows","FAR (%)"]
    rows = []
    for (b,s), c in counts.items():
        n = c["unauthorized"]
        rows.append([b,s,c["candidates"],c["paired"],n,c["false_allow"] if n else "N/A",
                     f"{100*c['false_allow']/n:.2f}" if n else "N/A"])
    with (args.output / "replay.csv").open("w",newline="") as f:
        csv.writer(f).writerows([header,*rows])
    (args.output / "replay.md").write_text("\n".join("| "+" | ".join(map(str,row))+" |" for row in [header,["---"]*len(header),*rows])+"\n")
    tex = [r"\begin{table}[t]\centering\small",r"\caption{Fixed-proposal replay. FAR is reported only for independently labeled unauthorized proposals.}",
           r"\begin{tabular}{llrr}\hline",r"Benchmark & Scope & Paired & FAR (\%) \\ \hline"]
    tex.extend(f"{r[0]} & {r[1]} & {r[3]} & {r[-1]} "+r"\\" for r in rows)
    (args.output / "replay.tex").write_text("\n".join([*tex,r"\hline\end{tabular}\end{table}"])+"\n")

if __name__ == "__main__":
    main()
