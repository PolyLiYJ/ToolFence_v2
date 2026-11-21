from toolfence.semantic.schema import ParameterSemantic, ToolSemantic

from experiments.real_benchmarks import config
from experiments.real_benchmarks.paper_profile import build_pinned_manifest
from experiments.real_benchmarks.runner_common import aggregate_rows, deterministic_subset, sha256_json


def test_deterministic_subset_is_stable():
    items = [{"id": str(i)} for i in range(200)]
    first = deterministic_subset(items, 20, config.SEED, key=lambda row: row["id"])
    second = deterministic_subset(items, 20, config.SEED, key=lambda row: row["id"])
    assert first == second
    assert len(first) == 20


def test_redacted_config_has_no_key_value():
    snapshot = config.redacted_config()
    assert snapshot["api_key_value"] == "<redacted>"
    assert snapshot["api_key_env"] == "GOOGLE_API_KEY"


def test_pinned_manifest_is_deterministic():
    catalog = [{
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]
    manifest = build_pinned_manifest(catalog)
    tool = manifest.by_name()["write_file"]
    assert isinstance(tool, ToolSemantic)
    assert isinstance(tool.parameters["path"], ParameterSemantic)
    assert tool.mutates_state is True
    assert sha256_json(catalog) == sha256_json(catalog)


def test_aggregate_rows():
    rows = [
        {"condition": "clean", "utility": True},
        {"condition": "attack", "utility": False, "attack_success": True, "judge_calls": 2},
        {"condition": "attack", "utility": True, "attack_success": False, "judge_calls": 1},
    ]
    result = aggregate_rows(rows)
    assert result["BU"] == 100.0
    assert result["U@A"] == 50.0
    assert result["ASR"] == 50.0
    assert result["judge_calls"] == 3
