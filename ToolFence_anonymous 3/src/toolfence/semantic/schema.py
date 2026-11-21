"""Typed policy objects for ToolFence-Semantic."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Effect = Literal[
    "read", "local_compute", "communication", "reservation", "financial",
    "credential_change", "external_write", "delete", "unknown",
]

# Primary operation is kept separate from effect.  A tool can have a secondary
# effect (for example, creating a calendar event may also send invitations),
# while the authenticated request normally names the primary operation ("add an
# event").  The controller uses operation only to ground side-effect evidence;
# runtime authorization still uses the trusted effect and parameter graph.
Operation = Literal[
    "observe", "compute", "communicate", "reserve", "create", "update",
    "delete", "financial", "credential_change", "unknown",
]

BindingKind = Literal[
    "literal", "normalized_literal", "template", "derived", "computed",
    "generated", "free",
]

CapabilityOrigin = Literal[
    "planner", "controller_dependency", "runtime_refinement",
    "controller_read_default", "runtime_capability_grant",
]


class ParameterSemantic(BaseModel):
    semantic_role: str
    authority_sensitive: bool = False


class ToolSemantic(BaseModel):
    name: str
    effect: Effect
    operation: Operation = "unknown"
    resource: str = "unknown"
    reversible: bool = False
    # Only a controller-pinned manifest may set these host-reviewed flags.
    reviewed_local_read: bool = False
    external_disclosure: bool = False
    mutates_state: bool = False
    parameters: Dict[str, ParameterSemantic] = Field(default_factory=dict)


class ToolManifest(BaseModel):
    tools: List[ToolSemantic]

    def by_name(self) -> Dict[str, ToolSemantic]:
        return {tool.name: tool for tool in self.tools}


class ParameterBinding(BaseModel):
    binding: BindingKind
    value: Optional[Any] = None
    # For normalized_literal/template, the controller stores authenticated text
    # that anchors the transformation.  These fields are audit evidence and are
    # never trusted merely because the policy model emitted them.
    evidence: str = ""
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    # Components supplied by deterministic normalization rather than verbatim
    # user text.  They are explicit in the audit record.
    inferred_components: List[str] = Field(default_factory=list)
    source_tools: List[str] = Field(default_factory=list)
    allowed_fields: List[str] = Field(default_factory=list)


class Capability(BaseModel):
    tool: str
    # Controller-owned ToolManifest is authoritative. The planner may omit this
    # field; validation always injects/overwrites it from the trusted manifest.
    effect: Optional[Effect] = None
    # Planner-produced operation/reuse/origin are never authoritative: the
    # controller overwrites them during validation.
    operation: Optional[Operation] = None
    reusable: bool = False
    origin: CapabilityOrigin = "planner"
    # evidence is retained for side-effect grounding.  The controller replaces
    # it with an exact authenticated-query slice before accepting a write.
    evidence: str = ""
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    max_calls: int = 1
    parameters: Dict[str, ParameterBinding] = Field(default_factory=dict)


class AuthorizationPlan(BaseModel):
    goal: str
    capabilities: List[Capability] = Field(default_factory=list)

    def by_tool(self) -> Dict[str, Capability]:
        """Backward-compatible first capability per tool."""
        result: Dict[str, Capability] = {}
        for capability in self.capabilities:
            result.setdefault(capability.tool, capability)
        return result

    def by_tool_all(self) -> Dict[str, List[Capability]]:
        result: Dict[str, List[Capability]] = {}
        for capability in self.capabilities:
            result.setdefault(capability.tool, []).append(capability)
        return result
