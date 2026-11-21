"""Compact stateful reference monitor for ToolFence-Core."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from toolfence.foundation.authorization import (
    Action,
    AuthorizationDecision,
    Provenance,
    ProvenanceEnvelope,
    StatefulAuthorityTracker,
)
from toolfence.foundation.blueprint import SessionBlueprint


@dataclass(frozen=True)
class CoreLease:
    lease_id: str
    action: Action
    issued_at: float
    expires_at: float


@dataclass(frozen=True)
class CoreDecision:
    allowed: bool
    violations: Tuple[str, ...] = ()
    lease: Optional[CoreLease] = None


class AtomicTrajectoryBudget:
    """Reserve trajectory authority before dispatch and roll it back on errors."""

    def __init__(self) -> None:
        self._committed: list[Action] = []
        self._pending: Dict[str, CoreLease] = {}
        self._lock = threading.RLock()

    def _live_pending(self, now: float) -> list[CoreLease]:
        expired = [key for key, lease in self._pending.items() if lease.expires_at <= now]
        for key in expired:
            self._pending.pop(key, None)
        return list(self._pending.values())

    @staticmethod
    def _hosts(action: Action) -> set[str]:
        from toolfence.foundation.authorization import CapabilityLedger

        return CapabilityLedger.action_hosts(action)

    @staticmethod
    def _shared_write(action: Action) -> bool:
        from toolfence.foundation.authorization import CapabilityLedger

        return CapabilityLedger.is_shared_write(action)

    @staticmethod
    def _optional_limit(blueprint: SessionBlueprint, name: str) -> Optional[int]:
        raw = blueprint.global_constraints.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    def reserve(
        self,
        action: Action,
        blueprint: SessionBlueprint,
        *,
        ttl_seconds: float = 30.0,
    ) -> CoreDecision:
        now = time.time()
        with self._lock:
            projected = [*self._committed, *(lease.action for lease in self._live_pending(now))]
            violations = []
            if len(projected) + 1 > int(blueprint.max_total_tool_calls):
                violations.append("budget:session_calls:exhausted")
            spec = blueprint.allowed_tools.get(action.tool)
            tool_calls = sum(item.tool == action.tool for item in projected)
            if spec is not None and tool_calls + 1 > int(spec.max_calls_per_session):
                violations.append(f"budget:{action.tool}:exhausted")
            host_limit = self._optional_limit(blueprint, "max_unique_hosts")
            if host_limit is not None:
                hosts = set().union(*(self._hosts(item) for item in projected)) if projected else set()
                if host_limit < 0 or len(hosts | self._hosts(action)) > host_limit:
                    violations.append("budget:max_unique_hosts:exhausted")
            write_limit = self._optional_limit(blueprint, "max_shared_writes")
            if write_limit is not None:
                writes = sum(self._shared_write(item) for item in projected)
                if write_limit < 0 or writes + int(self._shared_write(action)) > write_limit:
                    violations.append("budget:max_shared_writes:exhausted")
            if violations:
                return CoreDecision(False, tuple(dict.fromkeys(violations)))
            lease = CoreLease(
                lease_id=f"core-lease:{secrets.token_hex(12)}",
                action=action,
                issued_at=now,
                expires_at=now + ttl_seconds,
            )
            self._pending[lease.lease_id] = lease
            return CoreDecision(True, lease=lease)

    def commit(self, lease: CoreLease) -> None:
        with self._lock:
            stored = self._pending.pop(lease.lease_id, None)
            if stored is None or stored != lease or stored.expires_at <= time.time():
                raise PermissionError("Unknown, expired, or mismatched ToolFence-Core lease")
            self._committed.append(stored.action)

    def rollback(self, lease: CoreLease) -> None:
        with self._lock:
            self._pending.pop(lease.lease_id, None)

    @property
    def committed_count(self) -> int:
        with self._lock:
            return len(self._committed)


class CoreMonitor:
    """Enforce one immutable policy over the complete tool-call trajectory."""

    def __init__(self, blueprint: SessionBlueprint):
        self.tracker = StatefulAuthorityTracker(blueprint)
        self.budget = AtomicTrajectoryBudget()
        self._lock = threading.RLock()
        self._proposal_step = 0
        self._denials: list[dict[str, Any]] = []

    @property
    def blueprint(self) -> SessionBlueprint:
        return self.tracker.current

    def register_user_query(self, query: str) -> None:
        self.tracker.sources.register_user_query(query)

    def prepare(
        self,
        action: Any,
        envelope: ProvenanceEnvelope,
        *,
        lease_ttl_seconds: float = 30.0,
    ) -> CoreDecision:
        parsed = Action.from_obj(action)
        with self._lock:
            self._proposal_step += 1
            # Budgeting is handled by the atomic reservation below.  The legacy
            # ledger remains useful for secret-flow checks and source attribution.
            static: AuthorizationDecision = self.tracker.gate.evaluate(
                parsed,
                self.tracker.current,
                envelope,
                ledger=None,
                sources=self.tracker.sources,
            )
            if not static.allowed:
                violations = tuple(dict.fromkeys(static.violations))
                self._denials.append({
                    "step": self._proposal_step,
                    "action": parsed.to_dict(),
                    "violations": violations,
                })
                return CoreDecision(False, violations)
            reserved = self.budget.reserve(
                parsed, self.tracker.current, ttl_seconds=lease_ttl_seconds,
            )
            if not reserved.allowed:
                self._denials.append({
                    "step": self._proposal_step,
                    "action": parsed.to_dict(),
                    "violations": reserved.violations,
                })
            return reserved

    def commit(self, lease: CoreLease) -> None:
        self.budget.commit(lease)
        self.tracker.history.append({"type": "dispatch", "action": lease.action.to_dict()})

    def rollback(self, lease: CoreLease) -> None:
        self.budget.rollback(lease)

    def observe(self, value: Any, source: str) -> None:
        self.tracker.observe(value, Provenance.UNTRUSTED_TOOL, source=source)

    def metrics(self) -> Dict[str, Any]:
        return {
            "proposals": self._proposal_step,
            "denied_expansion_attempts": len(self._denials),
            "time_to_first_violation": (
                self._denials[0]["step"] if self._denials else None
            ),
            "realized_unauthorized_actions": 0,
            "committed_actions": self.budget.committed_count,
        }


class ContractionRepairer:
    """Only remove schema-unknown arguments; never invent or substitute values."""

    def propose(self, action: Action, blueprint: SessionBlueprint) -> Optional[Action]:
        spec = blueprint.allowed_tools.get(action.tool)
        if spec is None:
            return None
        retained = {
            name: value for name, value in action.arguments.items()
            if name in spec.parameter_rules
        }
        if retained == dict(action.arguments):
            return None
        return Action(action.tool, retained)
