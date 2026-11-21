"""
Cognitive Blueprint Definition and Management
Core component of intent-action decoupling architecture
"""

from typing import Dict, List, Optional, Any, Pattern, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
try:
    from pydantic import field_validator
except ImportError:  # Pydantic 1.x
    from pydantic import validator as field_validator
import json
import re
from enum import Enum


class ToolPermission(str, Enum):
    """Tool permission level"""
    ALLOWED = "allowed"
    DENIED = "denied"
    CONDITIONAL = "conditional"


class ParameterRule(BaseModel):
    """Parameter rule definition"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type")
    pattern: Optional[str] = Field(default=None, description="Regex pattern")
    min_length: Optional[int] = Field(default=None)
    max_length: Optional[int] = Field(default=None)
    allowed_values: Optional[List[str]] = Field(default=None)
    forbidden_values: Optional[List[str]] = Field(default=None)
    required: bool = Field(default=False, description="Whether the argument must be present")
    risk: str = Field(default="low", description="Risk tier: low, medium, or high")
    provenance: List[str] = Field(
        default_factory=list,
        description="Permitted value origins, for example USER_LITERAL or DELEGATED:bill.txt",
    )
    host_scope: List[str] = Field(
        default_factory=list,
        description="Permitted URL hosts. Subdomains are not implied unless prefixed with '*.'",
    )
    minimum: Optional[float] = Field(default=None)
    maximum: Optional[float] = Field(default=None)
    secret_policy: str = Field(
        default="deny_observed",
        description="Secret flow policy: deny_observed, provisioned_only, or allow",
    )
    declassify_types: List[str] = Field(
        default_factory=list,
        description="Low-capacity types that may be declassified (boolean or enum)",
    )
    
    def validate_value(self, value: Any) -> bool:
        """Validate if parameter value conforms to rules"""
        # Type check
        if self.type == "string" and not isinstance(value, str):
            return False
        if self.type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if self.type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return False
        if self.type == "boolean" and not isinstance(value, bool):
            return False
        if self.type == "array" and not isinstance(value, list):
            return False
        if self.type == "object" and not isinstance(value, dict):
            return False
        
        # Length check
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                return False
            if self.max_length and len(value) > self.max_length:
                return False
        
        # Regex match
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                return False
        
        # Allowed values list
        if self.allowed_values and value not in self.allowed_values:
            return False
        
        # Forbidden values list
        if self.forbidden_values and value in self.forbidden_values:
            return False

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                return False
            if self.maximum is not None and value > self.maximum:
                return False
        
        return True


class ToolBlueprint(BaseModel):
    """Single tool blueprint definition"""
    tool_id: str = Field(..., description="Tool unique identifier")
    permission: ToolPermission = Field(default=ToolPermission.DENIED)
    description: str = Field(default="", description="Tool functionality description")
    parameter_rules: Dict[str, ParameterRule] = Field(default_factory=dict)
    max_calls_per_session: int = Field(default=100)
    requires_confirmation: bool = Field(default=False)
    capability_domain: str = Field(default="general")
    egress: str = Field(default="scoped", description="Network policy: deny, scoped, or allow")
    secrets: str = Field(
        default="provisioned_only",
        description="Credential policy for this capability domain",
    )
    
    # Compatibility aliases for legacy code
    @property
    def allowed(self) -> bool:
        """Whether this tool is allowed (compatibility alias for permission)."""
        return self.permission == ToolPermission.ALLOWED
    
    @property
    def param_rules(self) -> Dict[str, ParameterRule]:
        """Alias for parameter_rules (compatibility)."""
        return self.parameter_rules
    
    def validate_call(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if tool call is legitimate
        
        Returns:
            (is_valid, error_message)
        """
        # CONDITIONAL is intentionally fail-closed here. A controller may
        # promote a conditional capability to ALLOWED after a separately
        # authenticated confirmation, but the tool boundary must never treat
        # the enum value itself as approval.
        if self.permission != ToolPermission.ALLOWED:
            return False, f"Tool '{self.tool_id}' is not allowed"
        
        missing = [
            name for name, rule in self.parameter_rules.items()
            if rule.required and name not in parameters
        ]
        if missing:
            return False, f"Missing required parameter(s): {', '.join(sorted(missing))}"

        # Validate parameters
        for param_name, param_value in parameters.items():
            if param_name not in self.parameter_rules:
                return False, f"Unknown parameter: {param_name}"
            
            rule = self.parameter_rules[param_name]
            if not rule.validate_value(param_value):
                return False, f"Invalid value for parameter '{param_name}': {param_value}"
        
        return True, ""


class SessionBlueprint(BaseModel):
    """
    Session-level cognitive blueprint
    Generated by Policy Architect, defines security constraints for the entire session
    """
    session_id: str = Field(..., description="Session unique identifier")
    user_intent: str = Field(default="", description="User's original intent summary")
    authorized_goals: List[str] = Field(default_factory=list)
    allowed_tools: Dict[str, ToolBlueprint] = Field(default_factory=dict)
    data_domains: List[str] = Field(default_factory=list)
    allowed_recipients: List[str] = Field(default_factory=list)
    context_constraints: Dict[str, Any] = Field(default_factory=dict)
    global_constraints: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: int = Field(default=300, description="Blueprint validity period")
    signature: Optional[str] = Field(default=None, description="Digital signature for tamper prevention")
    
    # Security policies
    max_total_tool_calls: int = Field(default=50)
    forbidden_patterns: List[str] = Field(default_factory=list)
    required_confirmations: List[str] = Field(default_factory=list)
    
    # Metadata (for compatibility)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('forbidden_patterns')
    def validate_patterns(cls, v):
        """Validate regex pattern legality"""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {pattern}, error: {e}")
        return v
    
    @property
    def authorized_tools(self) -> List[str]:
        """Get authorized tool list (compatibility property)"""
        return self.get_allowed_tool_names()
    
    @property
    def tool_blueprints(self) -> List[ToolBlueprint]:
        """Get tool blueprint list (compatibility property)"""
        return list(self.allowed_tools.values())
    
    @property
    def ttl(self) -> int:
        """Alias for ttl_seconds (compatibility)."""
        return self.ttl_seconds
    
    def is_tool_allowed(self, tool_id: str) -> bool:
        """Check if tool is allowed"""
        if tool_id not in self.allowed_tools:
            return False
        return self.allowed_tools[tool_id].permission == ToolPermission.ALLOWED
    
    def get_allowed_tool_names(self) -> List[str]:
        """Get all allowed tool names"""
        return [
            tool_id for tool_id, tool in self.allowed_tools.items()
            if tool.permission == ToolPermission.ALLOWED
        ]
    
    def validate_tool_call(self, tool_id: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool call"""
        if tool_id not in self.allowed_tools:
            return False, f"Tool '{tool_id}' not in allowed list"
        
        tool_blueprint = self.allowed_tools[tool_id]
        return tool_blueprint.validate_call(parameters)
    
    def check_content_safety(self, content: str) -> Tuple[bool, Optional[str]]:
        """
        Check if content contains forbidden patterns
        
        Returns:
            (is_safe, matched_pattern)
        """
        for pattern in self.forbidden_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, pattern
        return True, None
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return self.json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SessionBlueprint":
        """Deserialize from JSON"""
        return cls.parse_raw(json_str)


class BlueprintManager:
    """
    Blueprint manager
    Responsible for blueprint generation, validation, caching, and lifecycle management
    """
    
    def __init__(self):
        self._blueprint_cache: Dict[str, SessionBlueprint] = {}
        self._call_counters: Dict[str, Dict[str, int]] = {}
    
    def store_blueprint(self, blueprint: SessionBlueprint):
        """Store blueprint"""
        self._blueprint_cache[blueprint.session_id] = blueprint
        self._call_counters[blueprint.session_id] = {}
    
    def get_blueprint(self, session_id: str) -> Optional[SessionBlueprint]:
        """Get blueprint"""
        return self._blueprint_cache.get(session_id)
    
    def increment_call_counter(self, session_id: str, tool_id: str) -> int:
        """Increment tool call counter"""
        if session_id not in self._call_counters:
            self._call_counters[session_id] = {}
        
        current = self._call_counters[session_id].get(tool_id, 0)
        self._call_counters[session_id][tool_id] = current + 1
        return current + 1
    
    def check_rate_limit(self, session_id: str, tool_id: str, max_calls: int) -> bool:
        """Check if call limit is exceeded"""
        current = self._call_counters.get(session_id, {}).get(tool_id, 0)
        return current < max_calls
    
    def invalidate_blueprint(self, session_id: str):
        """Invalidate blueprint"""
        self._blueprint_cache.pop(session_id, None)
        self._call_counters.pop(session_id, None)


# Predefined secure blueprint templates
class BlueprintTemplates:
    """Common blueprint templates"""
    
    @staticmethod
    def web_search_only() -> SessionBlueprint:
        """Allow only web search"""
        return SessionBlueprint(
            session_id="template_web_search",
            user_intent="Safe web browsing and information retrieval",
            allowed_tools={
                "web_search": ToolBlueprint(
                    tool_id="web_search",
                    permission=ToolPermission.ALLOWED,
                    parameter_rules={
                        "query": ParameterRule(
                            name="query",
                            type="string",
                            pattern=r"^[a-zA-Z0-9\s\-_\.]{1,200}$",
                            min_length=1,
                            max_length=200
                        )
                    }
                ),
                "read_webpage": ToolBlueprint(
                    tool_id="read_webpage",
                    permission=ToolPermission.ALLOWED,
                    parameter_rules={
                        "url": ParameterRule(
                            name="url",
                            type="string",
                            pattern=r"^https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}.*$"
                        )
                    }
                )
            },
            forbidden_patterns=[
                r"(?:rm|del|remove)\s+-rf",
                r"curl\s+.*\|\s*bash",
                r"wget\s+.*\|\s*sh",
                r"eval\s*\(",
                r"exec\s*\("
            ]
        )
    
    @staticmethod
    def code_assistant_safe() -> SessionBlueprint:
        """Safe code assistant mode"""
        return SessionBlueprint(
            session_id="template_code_safe",
            user_intent="Safe code editing without system access",
            allowed_tools={
                "read_file": ToolBlueprint(
                    tool_id="read_file",
                    permission=ToolPermission.ALLOWED,
                    parameter_rules={
                        "path": ParameterRule(
                            name="path",
                            type="string",
                            pattern=r"^[\w\-/\\\.]+$",
                            forbidden_values=["/etc/passwd", "/etc/shadow", ".env"]
                        )
                    }
                ),
                "write_file": ToolBlueprint(
                    tool_id="write_file",
                    permission=ToolPermission.CONDITIONAL,
                    requires_confirmation=True,
                    parameter_rules={
                        "path": ParameterRule(
                            name="path",
                            type="string",
                            pattern=r"^[\w\-/\\\.]+\.(py|js|ts|java|cpp|c|h|md|txt|json|yaml)$"
                        ),
                        "content": ParameterRule(
                            name="content",
                            type="string",
                            max_length=100000
                        )
                    }
                ),
                "run_linter": ToolBlueprint(
                    tool_id="run_linter",
                    permission=ToolPermission.ALLOWED
                )
            },
            forbidden_patterns=[
                r"subprocess\.",
                r"os\.system",
                r"os\.popen",
                r"__import__",
                r"exec\s*\(",
                r"compile\s*\("
            ]
        )


class BlueprintValidator:
    """
    Blueprint validator
    
    Used to validate blueprint structural legality, signature, and integrity
    """
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
    def validate(self, blueprint: SessionBlueprint) -> bool:
        """
        Validate blueprint legality
        
        Args:
            blueprint: Blueprint to validate
            
        Returns:
            bool: Whether validation passed
        """
        self.validation_errors = []
        
        # Validate session ID
        if not blueprint.session_id:
            self.validation_errors.append("Session ID cannot be empty")
        
        # Validate user intent
        if not blueprint.user_intent:
            self.validation_errors.append("User intent cannot be empty")
        
        # Validate blueprint TTL
        if blueprint.ttl_seconds <= 0:
            self.validation_errors.append("TTL must be positive")
        
        # Validate forbidden pattern regex
        for pattern in blueprint.forbidden_patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                self.validation_errors.append(f"Invalid regex pattern: {pattern}, error: {e}")
        
        # Validate tool blueprints
        for tool_id, tool_blueprint in blueprint.allowed_tools.items():
            if tool_id != tool_blueprint.tool_id:
                self.validation_errors.append(
                    f"Tool ID mismatch: expected '{tool_id}', got '{tool_blueprint.tool_id}'"
                )
        
        # Validate signature
        if blueprint.signature is None:
            self.validation_errors.append("Blueprint signature is missing")
        
        return len(self.validation_errors) == 0
    
    def validate_signature(self, blueprint: SessionBlueprint, expected_signature: str) -> bool:
        """
        Validate blueprint signature
        
        Args:
            blueprint: Blueprint
            expected_signature: Expected signature
            
        Returns:
            bool: Whether signature matches
        """
        if blueprint.signature is None:
            return False
        return blueprint.signature == expected_signature
    
    def get_validation_errors(self) -> List[str]:
        """Get validation error list"""
        return self.validation_errors
    
    def reset(self):
        """Reset validator"""
        self.validation_errors = []


# Example usage
if __name__ == "__main__":
    # Create blueprint
    blueprint = SessionBlueprint(
        session_id="sess_001",
        user_intent="Search for Python tutorials",
        allowed_tools={
            "web_search": ToolBlueprint(
                tool_id="web_search",
                permission=ToolPermission.ALLOWED,
                parameter_rules={
                    "query": ParameterRule(
                        name="query",
                        type="string",
                        pattern=r"^[a-zA-Z0-9\s]{1,100}$"
                    )
                }
            )
        },
        forbidden_patterns=[r"rm\s+-rf", r"curl.*\|.*bash"]
    )
    
    print("Blueprint created:")
    print(blueprint.to_json())
    
    # Validate tool call
    is_valid, error = blueprint.validate_tool_call(
        "web_search",
        {"query": "Python tutorial"}
    )
    print(f"\nValidation result: {is_valid}, {error}")
    
    # Check forbidden patterns
    is_safe, pattern = blueprint.check_content_safety("rm -rf /")
    print(f"Content safety: {is_safe}, matched: {pattern}")
