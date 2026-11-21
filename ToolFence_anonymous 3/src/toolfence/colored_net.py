"""Small colored Petri net for task-scoped tool authority.

Places contain typed targets from the authenticated request. A successful
state-changing tool call consumes a target token; a denied or failed call does
not. This is an additional deterministic constraint on the hybrid blueprint,
not a source of authority from tool text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit


_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _origin(raw: str) -> str | None:
    """Canonical network origin; malformed URLs never create a matching token."""
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            return None
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
        return f"{parsed.scheme.casefold()}://{parsed.hostname.casefold()}:{port}"
    except ValueError:
        return None


class EffectClass(str, Enum):
    LOCAL_READ = "local_read"
    REMOTE_READ = "remote_read"  # a read can disclose its URL to a remote host
    EXTERNAL_MESSAGE = "external_message"
    FINANCIAL_TRANSFER = "financial_transfer"
    REMOTE_WRITE = "remote_write"
    OTHER = "other"


def effect_class(tool: str) -> EffectClass:
    if tool in {"get_webpage", "fetch_web_instance", "download_file_through_url"}:
        return EffectClass.REMOTE_READ
    if tool == "post_webpage":
        return EffectClass.REMOTE_WRITE
    if tool in {"send_email", "send_receive_email", "send_direct_message"}:
        return EffectClass.EXTERNAL_MESSAGE
    if tool == "send_money":
        return EffectClass.FINANCIAL_TRANSFER
    if tool.startswith(("read_", "get_", "search_", "list_", "view_")):
        return EffectClass.LOCAL_READ
    return EffectClass.OTHER


@dataclass(frozen=True)
class Token:
    place: str
    color: str
    source: str = "authenticated_request"


class ColoredAuthorityNet:
    """A conservative net: unknown task shapes defer to the existing guard.

    A transition is enabled only when all its target colors are in the same
    authenticated-request place. Financial effects also require an unspent
    token. The net never mints an authority token from a tool response.
    """

    def __init__(self, query: str):
        self.marking: set[Token] = set()
        self.consumed: set[Token] = set()
        for iban in _IBAN.findall(query):
            self.marking.add(Token("payment_target", iban.casefold()))
        for email in _EMAIL.findall(query):
            self.marking.add(Token("message_target", email.casefold()))
        for raw in _URL.findall(query):
            origin = _origin(raw.rstrip(".,;)]"))
            if origin:
                self.marking.add(Token("web_origin", origin))

    def snapshot(self) -> dict[str, Any]:
        return {
            "places": {place: sorted(t.color for t in self.marking if t.place == place)
                       for place in ("payment_target", "message_target", "web_origin")},
            "consumed_payments": sorted(t.color for t in self.consumed
                                        if t.place == "payment_target"),
        }

    def _targets(self, tool: str, args: Mapping[str, Any]) -> tuple[str, list[str]] | None:
        if tool == "send_money":
            value = args.get("recipient")
            return ("payment_target", [value.casefold()] if isinstance(value, str) else [])
        if tool == "send_email":
            targets: list[str] = []
            for field in ("recipients", "cc", "bcc"):
                value = args.get(field)
                if value is None:
                    continue
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    return ("message_target", [])
                targets.extend(v.casefold() for v in value)
            return ("message_target", targets)
        if tool in {"get_webpage", "fetch_web_instance", "download_file_through_url", "post_webpage"}:
            raw = args.get("url")
            origin = _origin(raw) if isinstance(raw, str) else None
            return ("web_origin", [origin] if origin else [])
        return None

    def decide(self, tool: str, args: Mapping[str, Any]) -> tuple[str, str, str] | None:
        target = self._targets(tool, args)
        if target is None:
            return None
        place, colors = target
        authorized = {t.color for t in self.marking if t.place == place}
        if not authorized:
            return None  # leave authority to the blueprint and runtime judge
        if not colors:
            return ("deny", f"{tool} has no valid {place}", "")
        if any(color not in authorized for color in colors):
            return ("deny", f"{tool} target is outside the authenticated request", "")
        if place == "payment_target" and any(
            Token(place, color) in self.consumed for color in colors
        ):
            return ("deny", "Payment target has already been used for this task", "")
        return None

    def observe(self, tool: str, args: Mapping[str, Any]) -> None:
        if tool != "send_money":
            return
        target = self._targets(tool, args)
        if target:
            place, colors = target
            self.consumed.update(Token(place, color) for color in colors)
