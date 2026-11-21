from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def deterministic_subset(
    items: Sequence[Any],
    size: int,
    seed: int,
    key: Callable[[Any], str] | None = None,
) -> list[Any]:
    ordered = sorted(items, key=(key or (lambda item: json.dumps(item, sort_keys=True))))
    if size >= len(ordered):
        return list(ordered)
    return random.Random(seed).sample(ordered, size)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows[str(row.get("key"))] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def add(self, key: str, row: Mapping[str, Any]) -> None:
        record = {"key": key, **dict(row)}
        self.rows[key] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    errors = [row for row in rows if row.get("error")]
    attacked = [row for row in valid if row.get("condition") == "attack"]
    clean = [row for row in valid if row.get("condition") == "clean"]
    return {
        "cases_attempted": len(rows),
        "cases_completed": len(valid),
        "errors": len(errors),
        "BU": pct(sum(bool(row.get("utility")) for row in clean), len(clean)),
        "U@A": pct(sum(bool(row.get("utility")) for row in attacked), len(attacked)),
        "ASR": pct(sum(bool(row.get("attack_success")) for row in attacked), len(attacked)),
        "denied": sum(int(row.get("denied", 0)) for row in valid),
        "judge_calls": sum(int(row.get("judge_calls", 0)) for row in valid),
        "runtime_seconds": round(sum(float(row.get("runtime_seconds", 0.0)) for row in rows), 3),
    }


def timed() -> Callable[[], float]:
    return time.perf_counter
