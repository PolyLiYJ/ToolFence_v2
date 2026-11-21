"""Cross-session cache for judge-approved capabilities.

A capability grant authorizes a *shape* (tool + effect + per-parameter
provenance constraints), not a concrete call. Cross-session reuse is restricted
to shapes whose parameters all came from authenticated user text. Derived,
computed, generated, free, and parameterless grants remain session-local.

Security: the cache stores only the safe capability *shape*, never argument
values. The per-call binding enforcement in SemanticMonitor._binding_violation
still runs on every concrete call, so a cached grant does not authorize
arbitrary values.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Tuple

from ..semantic.schema import Capability


def capability_signature(capability: Capability) -> Tuple:
    """Stable, value-free signature of a capability shape.

    Two capabilities with the same tool, effect, and parameter binding
    kinds/sources produce the same signature. Concrete argument values and
    evidence text are deliberately excluded.
    """
    parts = []
    for name in sorted(capability.parameters):
        binding = capability.parameters[name]
        parts.append((
            name,
            binding.binding,
            tuple(sorted(binding.source_tools)),
            tuple(sorted(binding.allowed_fields)),
        ))
    return (capability.tool, capability.effect, tuple(parts))


def cacheable_capability(capability: Capability) -> bool:
    """Return true only for shapes grounded wholly in authenticated user data.

    Runtime grant proposals represent authenticated-user parameters as literal
    bindings. Derived/computed/free/generated bindings may originate in
    untrusted observations or model choices and must remain session-local.
    Parameterless grants are also kept local because they carry no authenticated
    value tying the grant to the request.
    """
    return bool(capability.parameters) and all(
        binding.binding in {"literal", "normalized_literal"}
        and not binding.source_tools
        for binding in capability.parameters.values()
    )


class CapabilityGrantCache:
    """Process-level cache of granted capability shapes.

    Thread-safe. Hits avoid a judge call AND the grant-proposal construction:
    the cached Capability is injected straight into a new monitor's plan.
    """

    def __init__(self, *, revision: str = "legacy") -> None:
        if not revision:
            raise ValueError("A policy revision is required")
        self.revision = revision
        self._lock = Lock()
        # Store approval markers only. A Capability object contains concrete
        # literal values from the previous task and must not cross sessions.
        self._granted: set[Tuple] = set()
        self.hits = 0
        self.misses = 0
        self.puts = 0

    def get(self, signature: Tuple) -> bool:
        with self._lock:
            hit = (self.revision, signature) in self._granted
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            return hit

    def put(self, capability: Capability) -> bool:
        if not cacheable_capability(capability):
            return False
        sig = capability_signature(capability)
        with self._lock:
            self._granted.add((self.revision, sig))
            self.puts += 1
        return True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_shapes": len(self._granted),
                "policy_revision": self.revision,
                "hits": self.hits,
                "misses": self.misses,
                "puts": self.puts,
                "tools": sorted({sig[0] for _, sig in self._granted}),
            }

    def reset(self) -> None:
        with self._lock:
            self._granted.clear()
            self.hits = 0
            self.misses = 0
            self.puts = 0
