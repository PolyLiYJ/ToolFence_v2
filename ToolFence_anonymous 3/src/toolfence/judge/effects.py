"""Controller-owned effect classification for judge routing.

Only decides *routing*: read/local-compute calls may bypass the runtime judge
because they cannot complete an injection goal. This is never a security
decision by itself - anything not confidently classified as read is routed to
the judge, and every state-changing call is judged.

The semantic manifest (src/toolfence_semantic_v2) is the primary classifier.
It requires every tool to be classified exactly once, so a single missing or
duplicated entry discards the whole table; that failure mode left 24% of calls
unclassified in the first manifest-variant run, and read tools such as
read_channel_messages were then wrongly judged and denied. When the manifest is
unavailable or incomplete, a conservative verb heuristic on the controller-owned
tool name and description fills the gap.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

READ_PREFIXES = (
    "get_", "read_", "search_", "list_", "find_", "show_", "check_",
    "query_", "fetch_", "view_", "count_", "lookup_", "retrieve_",
    "get", "read", "search", "list", "find",
)

# A write verb anywhere in the description vetoes the read classification.
WRITE_HINTS = re.compile(
    r"\b(send|post|create|update|delete|remove|transfer|pay|book|reserve|"
    r"invite|publish|upload|write\s+to|modify|cancel|subscribe|grant|share|"
    r"append|insert|set\s+the)\w*\b",
    re.IGNORECASE,
)

READ_HINTS = re.compile(
    r"\b(return|returns|retrieve|retrieves|get|gets|list|lists|search|searches|"
    r"find|finds|read|reads|show|shows|display|fetch|look\s+up|obtain)\w*\b",
    re.IGNORECASE,
)

NO_MODIFICATION_HINTS = re.compile(
    r"(without\s+(modifying|changing|altering)|does\s+not\s+(modify|change)|"
    r"read[- ]only|no\s+side\s+effect)",
    re.IGNORECASE,
)


def _describes_write(description: str) -> bool:
    """True only when the description names a write as the tool's own action.

    A mere mention (for example "emails that were sent") must not veto a read
    classification, otherwise obvious readers such as get_received_emails are
    routed to the judge and wrongly denied.
    """
    if not description:
        return False
    head = description[:60]
    if re.match(r"^\s*(send|post|create|update|delete|remove|transfer|pay|book|"
                r"reserve|invite|publish|upload|modify|cancel|subscribe|grant|"
                r"share|append|insert|set)\b", head, re.IGNORECASE):
        return True
    if re.search(r"\bto\s+(send|post|create|update|delete|remove|transfer|pay|"
                 r"book|reserve|invite|publish|upload|modify|cancel|subscribe|"
                 r"grant|share|append|insert)\b", description, re.IGNORECASE):
        return True
    return False


def heuristic_effect(name: str, description: str = "") -> Optional[str]:
    """Return 'read' only when the tool is confidently observation-only."""
    lowered = (name or "").casefold().strip()
    if _describes_write(description):
        return None
    if lowered.startswith(READ_PREFIXES):
        return "read"
    text = f"{name} {description}"
    if READ_HINTS.search(text) and NO_MODIFICATION_HINTS.search(description or ""):
        return "read"
    return None


def classify_effects(
    catalog, manifest_effects: Optional[Mapping[str, str]] = None,
) -> dict:
    """Merge manifest classification with heuristic fallback."""
    effects: dict = {}
    for raw in catalog:
        function = raw.get("function", raw) if isinstance(raw, Mapping) else {}
        name = function.get("name")
        if not name:
            continue
        effect = (manifest_effects or {}).get(name)
        if effect is None:
            effect = heuristic_effect(name, function.get("description", ""))
        if effect is not None:
            effects[name] = effect
    return effects


def coverage_report(catalog, effects: Mapping[str, str]) -> dict:
    names = [
        (raw.get("function", raw) if isinstance(raw, Mapping) else {}).get("name")
        for raw in catalog
    ]
    names = [name for name in names if name]
    missing = [name for name in names if name not in effects]
    return {
        "tools": len(names),
        "classified": len(names) - len(missing),
        "missing": missing,
    }
