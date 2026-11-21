"""Read-only evaluation of one approved grant against a frozen pre-call state."""
from enum import Enum
import json


class Scope(str, Enum):
    TOOL = "tool"
    SOURCE = "source"
    STAGE = "stage"
    BUDGET = "budget"


class Incomplete(ValueError):
    pass


def require(mapping, key, kind):
    if not isinstance(mapping, dict) or type(mapping.get(key)) is not kind:
        raise Incomplete(key)
    return mapping[key]


def name(mapping, key):
    value = require(mapping, key, str)
    if not value.strip():
        raise Incomplete(key)
    return value


def decide(snapshot, scope):
    """Evidence records contain the host-observed value and its source/field ID."""
    scope = Scope(scope)
    origin = name(snapshot, "origin")
    if origin == "blueprint":
        return require(snapshot, "blueprint_allowed", bool)
    if origin != "approved_runtime_grant":
        raise Incomplete("approved runtime grant")
    grant = require(snapshot, "grant", dict)
    call = require(snapshot, "call", dict)
    state = require(snapshot, "state", dict)
    tool_ok = name(grant, "tool") == name(call, "tool")
    args = require(call, "arguments", dict)
    # 'bindings' is the host-declared full list of sensitive arguments.
    bindings = require(grant, "bindings", dict)
    evidence = require(state, "evidence", dict)
    sources = require(call, "sources", dict)
    source_ok = True
    for arg, source in bindings.items():
        if not isinstance(source, str) or not source:
            raise Incomplete("binding source")
        if arg not in args:
            source_ok = False
            continue
        observed = name(sources, arg)
        values = require(evidence, source, list)
        canonical = lambda v: json.dumps(v, sort_keys=True, allow_nan=False)
        if observed != source or not any(canonical(v) == canonical(args[arg]) for v in values):
            source_ok = False
    stage_ok = name(grant, "stage") == name(state, "stage")
    remaining = require(state, "remaining", int)
    if remaining < 0:
        raise Incomplete("negative remaining budget")
    checks = (tool_ok, source_ok, stage_ok, remaining > 0)
    return all(checks[:list(Scope).index(scope) + 1])
