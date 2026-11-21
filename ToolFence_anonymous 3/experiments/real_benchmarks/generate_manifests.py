from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config
from .manifest_schema import manifest_hash, validate_manifest
from .run_mcptox_100 import MCPTOX_PATH, parse_tools

OUT_DIR = Path(__file__).resolve().parent / "manifests"
OPERATIONAL = {
    "limit", "n", "page", "page_size", "offset", "format", "recurring", "dryRun",
    "include_archived", "recursive", "verbose", "excludePatterns",
}

AGENTDOJO_KIND = {
    "send_email": "communicate_email", "delete_email": "delete", "get_unread_emails": "read",
    "get_sent_emails": "read", "get_received_emails": "read", "get_draft_emails": "read",
    "search_emails": "read", "search_contacts_by_name": "read", "search_contacts_by_email": "read",
    "get_current_day": "read", "search_calendar_events": "read", "get_day_calendar_events": "read",
    "create_calendar_event": "create", "cancel_calendar_event": "delete",
    "reschedule_calendar_event": "update", "add_calendar_event_participants": "update",
    "append_to_file": "write", "search_files_by_filename": "read", "create_file": "create",
    "delete_file": "delete", "get_file_by_id": "read", "list_files": "read",
    "share_file": "communicate", "search_files": "read",
    "get_channels": "read", "add_user_to_channel": "update", "read_channel_messages": "read",
    "read_inbox": "read", "send_direct_message": "communicate", "send_channel_message": "communicate",
    "get_users_in_channel": "read", "invite_user_to_slack": "invite", "remove_user_from_slack": "delete",
    "get_webpage": "web_read", "post_webpage": "web_post",
    "get_iban": "read", "send_money": "financial", "schedule_transaction": "financial",
    "update_scheduled_transaction": "update", "get_balance": "read",
    "get_most_recent_transactions": "read", "get_scheduled_transactions": "read",
    "read_file": "read", "get_user_info": "read", "update_password": "account_update",
    "update_user_info": "account_update",
    "get_user_information": "read", "get_all_hotels_in_city": "read", "get_hotels_prices": "read",
    "get_rating_reviews_for_hotels": "read", "get_hotels_address": "read",
    "get_all_restaurants_in_city": "read", "get_cuisine_type_for_restaurants": "read",
    "get_restaurants_address": "read", "get_rating_reviews_for_restaurants": "read",
    "get_dietary_restrictions_for_all_restaurants": "read",
    "get_contact_information_for_restaurants": "read", "get_price_for_restaurants": "read",
    "check_restaurant_opening_hours": "read", "get_all_car_rental_companies_in_city": "read",
    "get_car_types_available": "read", "get_rating_reviews_for_car_rental": "read",
    "get_car_fuel_options": "read", "get_car_rental_address": "read", "get_car_price_per_day": "read",
    "reserve_hotel": "reserve", "reserve_car_rental": "reserve", "reserve_restaurant": "reserve",
    "get_flight_information": "read",
}

KIND_SEMANTICS = {
    "read": ("read", "observe", "benchmark_data", False, False, False, False, "user_goal"),
    "web_read": ("read", "observe", "web", False, False, False, False, "user_literal"),
    "local_compute": ("local_compute", "compute", "benchmark_data", False, False, False, False, "user_goal"),
    "communicate": ("communication", "communicate", "message", True, True, False, True, "user_literal"),
    "communicate_email": ("communication", "communicate", "email", True, True, False, True, "user_literal"),
    "invite": ("communication", "communicate", "collaboration", True, True, False, True, "user_literal"),
    "web_post": ("external_write", "update", "web", True, True, False, True, "user_literal"),
    "create": ("external_write", "create", "benchmark_resource", True, True, False, True, "user_literal"),
    "write": ("external_write", "update", "benchmark_resource", True, True, False, True, "user_literal"),
    "update": ("external_write", "update", "benchmark_resource", True, True, False, True, "user_literal"),
    "delete": ("delete", "delete", "benchmark_resource", True, True, False, True, "previous_read_output"),
    "financial": ("financial", "financial", "payment", True, True, False, True, "user_literal"),
    "reserve": ("financial", "reserve", "reservation", True, True, False, True, "user_literal"),
    "account_update": ("account_update", "credential_change", "account", True, False, False, True, "user_literal"),
}


def make_tool(name: str, kind: str, schema: dict, description: str = "", host_adapter: str = "benchmark.sandbox.invoke") -> dict:
    effect, operation, resource, reversible, reviewed_local_read, external_disclosure, mutates_state, default_grounding = KIND_SEMANTICS[kind]
    properties = schema.get("properties", {}) if isinstance(schema, MappingLike) else {}
    params = {}
    for key in properties:
        if key in OPERATIONAL:
            params[key] = {"semantic_role": f"operational:{key}", "authority_sensitive": False, "grounding_rule": "operational_default"}
        else:
            sensitive = mutates_state or effect in {"communication", "financial", "account_update"}
            params[key] = {
                "semantic_role": f"reviewed:{key}",
                "authority_sensitive": sensitive,
                "grounding_rule": "user_literal" if sensitive else default_grounding,
            }
    return {
        "name": name,
        "description": description or name,
        "effect": effect,
        "operation": operation,
        "resource": resource,
        "reversible": reversible,
        "reviewed_local_read": reviewed_local_read,
        "external_disclosure": external_disclosure,
        "mutates_state": mutates_state,
        "parameters": params,
        "host_adapter": host_adapter,
        "allowed_in_attack_context": False,
        "grounding_rule": default_grounding,
        "schema": schema,
    }


MappingLike = dict


def agentdojo_schema(tool) -> dict:
    schema = tool.parameters
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    return {"type": "object", "properties": {}}


def build_agentdojo() -> dict:
    from agentdojo.task_suite.load_suites import get_suite

    tools = {}
    contexts = {}
    for suite_name in ("workspace", "slack", "banking", "travel"):
        suite = get_suite("v1", suite_name)
        names = []
        for tool in suite.tools:
            name = tool.name
            kind = AGENTDOJO_KIND.get(name)
            if kind is None:
                continue
            tools[name] = make_tool(name, kind, agentdojo_schema(tool), tool.description or "")
            names.append(name)
        contexts[suite_name] = {"tools": names}
    missing = sorted({t.name for s in ("workspace", "slack", "banking", "travel") for t in get_suite("v1", s).tools} - set(tools))
    return {
        "manifest_version": "1.0.0",
        "benchmark": "agentdojo",
        "policy_revision": "reviewed-manifest-v1-agentdojo",
        "fail_closed": True,
        "unknown_tool_policy": "deny",
        "source_provenance": {"source": "AgentDojo v1 suite.tools", "excluded_unreviewed": missing},
        "contexts": contexts,
        "tools": tools,
        "denylist": missing,
    }


def build_asb() -> dict:
    sys.path.insert(0, "/home/yjli/Agent/Blueprinting")
    from benchmarks.asb_memory import ASBMemoryPoisoningBenchmark

    bench = ASBMemoryPoisoningBenchmark(source_root=config.DATA_ROOT / "asb_orig", task_limit_per_agent=2)
    normal = sorted({tool.name for case in bench.cases for tool in case.normal_tools})
    attacks = sorted({case.attack_tool.name for case in bench.cases})
    tools = {
        name: make_tool(name, "read", {"type": "object", "properties": {}}, f"ASB reviewed normal tool: {name}")
        for name in normal
    }
    return {
        "manifest_version": "1.0.0",
        "benchmark": "asb",
        "policy_revision": "reviewed-manifest-v1-asb",
        "fail_closed": True,
        "unknown_tool_policy": "deny",
        "source_provenance": {"source": "official ASB agent_task/all_normal_tools/all_attack_tools", "cases": len(bench.cases)},
        "contexts": {"default": {"tools": normal}},
        "tools": tools,
        "denylist": attacks,
    }


def mcptox_kind(name: str) -> str:
    n = name.lower()
    if any(token in n for token in ("delete", "remove", "cancel", "void", "reset", "sabotage")):
        return "delete"
    if any(token in n for token in ("send", "post", "put_", "create", "write", "update", "upload", "charge", "subscribe", "migrate", "invite", "reply", "react")):
        return "update"
    if any(token in n for token in ("reserve", "book", "pay", "payment", "purchase")):
        return "financial"
    if any(token in n for token in ("get", "list", "read", "search", "retrieve", "extract", "query", "crawl", "scrape", "screenshot", "convert", "count", "analyze", "check", "distance", "weather", "history", "standings", "leaderboard")):
        return "read"
    return "local_compute"


def build_mcptox() -> dict:
    data = json.loads(MCPTOX_PATH.read_text(encoding="utf-8"))
    tools: dict[str, dict] = {}
    contexts: dict[str, dict] = {}
    denylist: set[str] = set()
    for server_name, server in data["servers"].items():
        clean = parse_tools(server.get("clean_system_promot", ""))
        names = []
        for name in server.get("tool_names", []):
            item = clean.get(name, {"function": {"name": name, "parameters": {"type": "object", "properties": {}}}})
            fn = item["function"]
            tools[name] = make_tool(
                name, mcptox_kind(name), fn.get("parameters", {"type": "object", "properties": {}}),
                fn.get("description", ""), "benchmark.sandbox.invoke",
            )
            names.append(name)
        contexts[server_name] = {"tools": names}
        for malicious in server.get("malicious_instance", []):
            for entry in malicious.get("datas", []):
                poisoned = parse_tools(entry.get("system", ""))
                denylist.update(set(poisoned) - set(clean))
    return {
        "manifest_version": "1.0.0",
        "benchmark": "mcptox",
        "policy_revision": "reviewed-manifest-v1-mcptox",
        "fail_closed": True,
        "unknown_tool_policy": "deny",
        "source_provenance": {
            "source": "MCPTox clean_system_promot/tool_names only; poisoned-only tools excluded",
            "path": str(MCPTOX_PATH),
            "servers": len(data["servers"]),
        },
        "contexts": contexts,
        "tools": tools,
        "denylist": sorted(denylist),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for benchmark, builder in (
        ("agentdojo", build_agentdojo), ("asb", build_asb), ("mcptox", build_mcptox)
    ):
        manifest = builder()
        validate_manifest(manifest)
        path = OUT_DIR / f"{benchmark}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{benchmark}: {path} tools={len(manifest['tools'])} contexts={len(manifest['contexts'])} deny={len(manifest['denylist'])} sha256={manifest_hash(manifest)}")


if __name__ == "__main__":
    main()
