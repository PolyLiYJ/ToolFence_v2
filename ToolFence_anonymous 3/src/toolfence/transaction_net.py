"""Controller-owned bounded colored net. Host adapters are the trust boundary.

This kernel is opt-in; legacy runtimes do not silently acquire new authority.
Only the authenticated control plane may issue contracts or ingest attestations.
Model/tool text must never be passed to those methods as trusted input.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import sqlite3
from typing import Callable, Mapping
from uuid import UUID


def identifier(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Identifiers must be canonical UUIDs")
    return value


@dataclass(frozen=True)
class ObjectRef:
    namespace: str
    kind: str
    key: str
    version: str

    def encode(self) -> str:
        if not all(isinstance(x, str) and x for x in (self.namespace, self.kind, self.key, self.version)):
            raise ValueError("Object references require namespace, kind, key and version")
        return json.dumps([self.namespace, self.kind, self.key, self.version], separators=(",", ":"))


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    budget_id: str
    tool: str
    bindings: Mapping[str, str | int]
    max_calls: int = 1
    max_units: int = 1


@dataclass(frozen=True)
class Receipt:
    operation_id: str
    outcome: str  # committed, not_applied, unknown; supplied by trusted transport


class RestrictedAdapter:
    """Pinned host-side adapter; names/descriptions from MCP cannot replace it."""
    def __init__(self, tool: str, fields: set[str], cost_field: str | None = None,
                 evidence_kind: str | None = None):
        self.tool, self.fields = tool, frozenset(fields)
        self.cost_field, self.evidence_kind = cost_field, evidence_kind

    def validate(self, bindings: Mapping, arguments: Mapping) -> int:
        if set(arguments) != self.fields or set(bindings) != self.fields:
            raise PermissionError("Exact adapter argument set required")
        if any(type(arguments[k]) is not type(bindings[k]) or arguments[k] != bindings[k] for k in self.fields):
            raise PermissionError("Invocation differs from authenticated instance bindings")
        for key in self.fields - ({self.cost_field} if self.cost_field else set()):
            if not isinstance(arguments[key], str) or not arguments[key]:
                raise ValueError("Reference and digest fields must be nonempty strings")
        for key in self.fields:
            if key.endswith("_ref"):
                try:
                    if ObjectRef(*json.loads(arguments[key])).encode() != arguments[key]:
                        raise ValueError("Noncanonical reference")
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid typed object reference") from exc
            if key.endswith("_digest"):
                value = arguments[key]
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Digests must be lowercase SHA-256 hex")
        if "currency" in arguments and (len(arguments["currency"]) != 3 or not arguments["currency"].isascii() or not arguments["currency"].isupper() or not arguments["currency"].isalpha()):
            raise ValueError("Currency requires an uppercase three-letter code")
        cost = arguments[self.cost_field] if self.cost_field else 1
        if type(cost) is not int or cost <= 0:
            raise ValueError("Costs must be positive integer minor units")
        return cost


PAYMENT = RestrictedAdapter("payment.transfer", {"account_ref", "amount_minor", "currency"}, "amount_minor")
ORDER_OTP = RestrictedAdapter("order.verify", {"order_ref", "otp_digest"}, evidence_kind="order_otp")
DOCUMENT_SEND = RestrictedAdapter("document.send", {"document_ref", "recipient_ref", "content_digest"})
# Only for the benchmark's inert, in-process Sandbox. Production integrations
# must supply effect-specific adapters rather than treating this as an RPC grant.
SANDBOX_OPERATION = RestrictedAdapter("benchmark.sandbox.invoke", {"operation_ref", "arguments_digest"})
ADAPTERS = {a.tool: a for a in (PAYMENT, ORDER_OTP, DOCUMENT_SEND, SANDBOX_OPERATION)}


class TransactionNet:
    def __init__(self, path: str):
        self.path = path
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budgets(id TEXT PRIMARY KEY, task TEXT NOT NULL,
                    calls INTEGER NOT NULL, units INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS contracts(task TEXT, tool TEXT, budget TEXT,
                    bindings TEXT NOT NULL, PRIMARY KEY(task,tool));
                CREATE TABLE IF NOT EXISTS evidence(task TEXT, kind TEXT, payload TEXT,
                    PRIMARY KEY(task,kind,payload));
                CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY, task TEXT,
                    tool TEXT, budget TEXT, arguments TEXT, units INTEGER, state TEXT);
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def canonical(value: Mapping) -> str:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def issue(self, contract: TaskContract):
        """Trusted host only. Budget IDs cannot be renewed or shared across tasks."""
        identifier(contract.task_id); identifier(contract.budget_id)
        adapter = ADAPTERS[contract.tool]
        adapter.validate(contract.bindings, contract.bindings)
        if type(contract.max_calls) is not int or type(contract.max_units) is not int or min(contract.max_calls, contract.max_units) <= 0:
            raise ValueError("Positive bounded budget required")
        with self.transaction() as db:
            db.execute("INSERT INTO budgets VALUES(?,?,?,?)", (contract.budget_id, contract.task_id, contract.max_calls, contract.max_units))
            db.execute("INSERT INTO contracts VALUES(?,?,?,?)", (contract.task_id, contract.tool, contract.budget_id, self.canonical(contract.bindings)))

    def ingest(self, task: str, kind: str, payload: Mapping, verify: Callable[[str, str, Mapping], bool]):
        """verify is a controller-pinned attestor, never supplied by an agent."""
        identifier(task)
        frozen = json.loads(self.canonical(payload))
        if not verify(task, kind, frozen):
            raise PermissionError("Evidence is not attested")
        with self.transaction() as db:
            if not db.execute("SELECT 1 FROM contracts WHERE task=?", (task,)).fetchone():
                raise PermissionError("Unknown task")
            db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?)", (task, kind, self.canonical(frozen)))

    def reserve(self, operation: str, task: str, tool: str, arguments: Mapping,
                *, grant_id: str | None = None, actor: str | None = None) -> str:
        identifier(operation); identifier(task)
        encoded = self.canonical(arguments)
        frozen = json.loads(encoded)
        with self.transaction() as db:
            old = db.execute("SELECT * FROM operations WHERE id=?", (operation,)).fetchone()
            if old:
                if (old['task'], old['tool'], old['arguments']) != (task, tool, encoded):
                    raise PermissionError("Idempotency key reused with a different invocation")
                return old['state']
            contract = db.execute("SELECT * FROM contracts WHERE task=? AND tool=?", (task, tool)).fetchone()
            if not contract:
                raise PermissionError("No authenticated task contract")
            self._reserve_guard(db, contract, frozen, operation, grant_id, actor)
            adapter = ADAPTERS[tool]
            cost = adapter.validate(json.loads(contract['bindings']), frozen)
            if adapter.evidence_kind and not db.execute("SELECT 1 FROM evidence WHERE task=? AND kind=? AND payload=?", (task, adapter.evidence_kind, encoded)).fetchone():
                raise PermissionError("Missing order-bound OTP attestation")
            changed = db.execute("UPDATE budgets SET calls=calls-1, units=units-? WHERE id=? AND task=? AND calls>=1 AND units>=?", (cost, contract['budget'], task, cost)).rowcount
            if changed != 1:
                raise PermissionError("Budget exhausted or already reserved")
            db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?)", (operation, task, tool, contract['budget'], encoded, cost, "reserved"))
        return "reserved"

    def _settle(self, operation: str, source: tuple[str, ...], target: str, refund: bool = False):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM operations WHERE id=?", (operation,)).fetchone()
            if row is None or row['state'] not in source:
                raise PermissionError("Disabled net transition")
            self._settle_guard(db, row, target)
            if refund:
                db.execute("UPDATE budgets SET calls=calls+1, units=units+? WHERE id=?", (row['units'], row['budget']))
            db.execute("UPDATE operations SET state=? WHERE id=?", (target, operation))
            return dict(row)

    def _reserve_guard(self, db, contract, arguments, operation, grant_id, actor):
        if grant_id is not None or actor is not None:
            raise PermissionError("Typed grants require GuardedTransactionNet")

    def _settle_guard(self, db, operation, target):
        pass

    def commit(self, receipt: Receipt):
        """Host attested receipt only; agents cannot submit a raw success flag."""
        if not isinstance(receipt, Receipt) or receipt.outcome != "committed":
            raise PermissionError("Committed receipt required")
        self._settle(receipt.operation_id, ("dispatched",), "committed")

    def unknown(self, operation: str):
        self._settle(operation, ("dispatched",), "unknown")

    def abort(self, operation: str):
        self._settle(operation, ("reserved",), "aborted", refund=True)

    def dispatch(self, operation: str, transport: Callable[[str, str, Mapping], Receipt]) -> str:
        row = self._settle(operation, ("reserved",), "dispatched")
        try:
            receipt = transport(operation, row['tool'], json.loads(row['arguments']))
            if not isinstance(receipt, Receipt) or receipt.operation_id != operation:
                raise ValueError("Receipt is not bound to this operation")
        except BaseException:
            self._settle(operation, ("dispatched",), "unknown")
            raise
        return self._receipt(receipt, ("dispatched",))

    def _receipt(self, receipt: Receipt, source: tuple[str, ...]) -> str:
        target = {"committed": "committed", "not_applied": "aborted", "unknown": "unknown"}.get(receipt.outcome, "unknown")
        self._settle(receipt.operation_id, source, target, refund=target == "aborted")
        return target

    def reconcile(self, operation: str, query_status: Callable[[str], Receipt]) -> str:
        # A process crash after dispatch leaves dispatched; never assume abort.
        receipt = query_status(operation)
        if not isinstance(receipt, Receipt) or receipt.operation_id != operation:
            raise PermissionError("Reconciliation receipt mismatch")
        return self._receipt(receipt, ("unknown", "dispatched"))


@dataclass(frozen=True)
class HostToolBinding:
    task_id: str
    adapter: str
    encode: Callable[[Mapping], Mapping]
    attest_receipt: Callable[[str, object], Receipt]
    raw_fields: frozenset[str]


class TransactionBoundary:
    """Final host dispatch boundary, shared by all policy/judge fast paths.

    bindings and read_guard come from trusted integration code. read_guard must
    check bounded read arguments/effects (including disclosure), not tool names.
    No registered binding means no effect dispatch.
    """
    def __init__(self, inner, net: TransactionNet, bindings: Mapping[str, HostToolBinding],
                 read_guard: Callable[[str, Mapping], bool] | None = None):
        self.inner, self.net = inner, net
        self.bindings = dict(bindings)
        self.read_guard = read_guard
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _authorization(self, binding, canonical, raw_arguments=None):
        return {}

    def run_function(self, env, function, kwargs, raise_on_error=False):
        from uuid import uuid4
        # Snapshot once, so validation and actual dispatch see identical data.
        args = json.loads(self.net.canonical(kwargs))
        binding = self.bindings.get(function)
        if binding is None:
            if self.read_guard is not None and self.read_guard(function, args):
                return self.inner.run_function(env, function, args, raise_on_error=raise_on_error)
            message = "not_evaluated: missing trusted tool contract adapter"
            self.decisions.append({'tool': function, 'state': 'denied', 'reason': message})
            if raise_on_error:
                raise PermissionError(message)
            return '', message
        operation = str(uuid4())
        try:
            if set(args) != binding.raw_fields:
                raise PermissionError("Invocation has missing or unregistered raw arguments")
            canonical = binding.encode(args)
            authorization = self._authorization(binding, canonical, raw_arguments=args)
            self.net.reserve(operation, binding.task_id, binding.adapter, canonical, **authorization)
        except (ValueError, PermissionError, KeyError, TypeError) as exc:
            self.decisions.append({'tool': function, 'state': 'denied', 'reason': str(exc)})
            if raise_on_error:
                raise
            return '', str(exc)
        results = []
        entered_transport = False
        def transport(op, _, canonical_args):
            nonlocal entered_transport
            entered_transport = True
            # encode() is host code; canonical_args are the exact reserved values.
            result = self.inner.run_function(env, function, args, raise_on_error=raise_on_error)
            results.append(result)
            return binding.attest_receipt(op, result)
        try:
            state = self.net.dispatch(operation, transport)
        except BaseException as exc:
            if not entered_transport:
                self.net.abort(operation)
                self.decisions.append({'tool': function, 'operation': operation, 'state': 'denied', 'reason': str(exc)})
                if raise_on_error or not isinstance(exc, Exception):
                    raise
                return '', str(exc)
            self.decisions.append({'tool': function, 'operation': operation, 'state': 'unknown'})
            if raise_on_error or not isinstance(exc, Exception):
                raise
            return '', 'Outcome unknown; reconcile operation ' + operation
        self.decisions.append({'tool': function, 'operation': operation, 'state': state})
        if state == 'unknown':
            return '', 'Outcome unknown; reconcile operation ' + operation
        return results[0]
