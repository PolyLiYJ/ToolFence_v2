"""Persistent guarded authorization net over the transactional execution kernel.

All configure/advance/activate/revoke/review APIs are HOST control-plane APIs.
Never expose them as agent tools. The refinement channel deliberately widens
the historical activation principle: model output may now PROPOSE new exact
contracts and transition rules, but every proposal shape requires one host
review before it applies, budgets are minted from host-supplied numbers only,
and reuse of an approved shape on another task additionally requires host
value attestation of that task's fresh argument values. A reviewer can only
approve or veto; it can never edit values, stages, evidence or budgets.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Mapping
from uuid import uuid4

from .transaction_net import ADAPTERS, TransactionNet, TransactionBoundary, identifier


@dataclass(frozen=True)
class TransitionRule:
    stage: str
    next_stage: str
    evidence_kind: str | None = None  # attested payload must equal canonical action


@dataclass(frozen=True)
class EvidenceStep:
    stage: str
    next_stage: str
    kind: str
    payload: dict


@dataclass(frozen=True)
class RefinementProposal:
    """Untrusted agent/LLM proposal for a new exact contract plus rule.

    requested_* budgets are advisory review data only; the host mints the
    actual budget. bindings are exact adapter-validated argument values.
    A proposal is pending review data: by itself it grants no authority.
    """
    task: str
    tool: str
    bindings: Mapping
    stage: str
    next_stage: str
    evidence_kind: str | None = None
    requested_calls: int = 1
    requested_units: int = 1
    note: str = ''


class GuardedTransactionNet(TransactionNet):
    """Finite workflow places, exact colored evidence, and attenuated grants.

    Each task has one workflow token. Reserve locks it; committed receipts move
    it; abort releases it; ambiguous outcomes retain it. All are ledger-atomic.
    """
    def __init__(self, path, *, clock=time.time):
        super().__init__(path)
        self.clock = clock
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS guarded_tasks(
                    task TEXT PRIMARY KEY, stage TEXT, holder TEXT, revision TEXT,
                    expires REAL, revoked INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS guarded_rules(
                    task TEXT, tool TEXT, rule TEXT, PRIMARY KEY(task,tool));
                CREATE TABLE IF NOT EXISTS guarded_steps(
                    task TEXT, name TEXT, step TEXT, PRIMARY KEY(task,name));
                CREATE TABLE IF NOT EXISTS guarded_grants(
                    id TEXT PRIMARY KEY, task TEXT, tool TEXT, actor TEXT, expires REAL,
                    parent TEXT, revoked INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS guarded_reservations(
                    operation TEXT PRIMARY KEY, grant_id TEXT, actor TEXT, next_stage TEXT);
                CREATE TABLE IF NOT EXISTS guarded_templates(
                    key TEXT PRIMARY KEY, shape TEXT);
                CREATE TABLE IF NOT EXISTS guarded_refinements(
                    id TEXT PRIMARY KEY, task TEXT, tool TEXT,
                    proposal TEXT, state TEXT, route TEXT);
                CREATE TABLE IF NOT EXISTS guarded_refinement_templates(
                    key TEXT PRIMARY KEY, shape TEXT);
            ''')

    def _expiry(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= self.clock():
            raise ValueError('Expiry must be finite and in the future')
        return value

    def configure(self, task, *, initial_stage, rules, steps=None, revision, expires):
        """Pin host-reviewed rules to already issued exact task contracts, once."""
        identifier(task)
        self._expiry(expires)
        if not initial_stage or not revision or not rules:
            raise ValueError('Initial stage, revision and rules required')
        rules = {tool: asdict(rule) for tool, rule in rules.items()}
        steps = {name: asdict(step) for name, step in (steps or {}).items()}
        for rule in [*rules.values(), *steps.values()]:
            if not all(isinstance(rule[k], str) and rule[k] for k in ('stage', 'next_stage')):
                raise ValueError('Nonempty stage names required')
        with self.transaction() as db:
            tools = {r['tool'] for r in db.execute('SELECT tool FROM contracts WHERE task=?', (task,))}
            if tools != rules.keys():
                raise PermissionError('Rules must cover exactly the host task contracts')
            if db.execute('SELECT 1 FROM operations WHERE task=?', (task,)).fetchone():
                raise PermissionError('Cannot configure a task after execution begins')
            db.execute('INSERT INTO guarded_tasks(task,stage,revision,expires) VALUES(?,?,?,?)', (task, initial_stage, revision, expires))
            db.executemany('INSERT INTO guarded_rules VALUES(?,?,?)', [(task, t, self.canonical(r)) for t, r in rules.items()])
            db.executemany('INSERT INTO guarded_steps VALUES(?,?,?)', [(task, n, self.canonical(r)) for n, r in steps.items()])

    def _scope(self, db, task):
        scope = db.execute('SELECT * FROM guarded_tasks WHERE task=?', (task,)).fetchone()
        if scope is None or scope['revoked'] or scope['expires'] <= self.clock():
            raise PermissionError('Task scope missing, revoked or expired')
        return scope

    def _grant(self, db, grant_id, task, tool, actor):
        seen = set()
        while grant_id:
            if grant_id in seen:
                raise PermissionError('Cyclic grant ancestry')
            seen.add(grant_id)
            grant = db.execute('SELECT * FROM guarded_grants WHERE id=?', (grant_id,)).fetchone()
            if (grant is None or grant['task'] != task or grant['tool'] != tool
                    or grant['revoked'] or grant['expires'] <= self.clock()):
                raise PermissionError('Grant missing, revoked, expired or outside task')
            if len(seen) == 1 and grant['actor'] != actor:
                raise PermissionError('Grant belongs to another agent')
            grant_id = grant['parent']

    def advance(self, task, name):
        """Fire a host-pinned evidence transition; observations alone cannot fire."""
        with self.transaction() as db:
            scope = self._scope(db, task)
            row = db.execute('SELECT step FROM guarded_steps WHERE task=? AND name=?', (task, name)).fetchone()
            if row is None:
                raise PermissionError('Unknown evidence transition')
            step = json.loads(row['step'])
            if scope['holder'] or scope['stage'] != step['stage']:
                raise PermissionError('Workflow token unavailable or wrong stage')
            if not db.execute('SELECT 1 FROM evidence WHERE task=? AND kind=? AND payload=?',
                              (task, step['kind'], self.canonical(step['payload']))).fetchone():
                raise PermissionError('Missing exact attested workflow evidence')
            db.execute('UPDATE guarded_tasks SET stage=? WHERE task=?', (step['next_stage'], task))

    def activate(self, task, tool, *, actor, expires, parent=None):
        """Host-only attenuation: exact parent contract, SAME budget, bounded TTL.

        A delegated child can change actor only through this trusted API. Revoking
        any ancestor disables all descendants. No new budget is inserted here.
        """
        self._expiry(expires)
        if not isinstance(actor, str) or not actor:
            raise ValueError('Authenticated actor identity required')
        grant_id = str(uuid4())
        with self.transaction() as db:
            scope = self._scope(db, task)
            if expires > scope['expires'] or not db.execute('SELECT 1 FROM guarded_rules WHERE task=? AND tool=?', (task, tool)).fetchone():
                raise PermissionError('Grant exceeds task scope')
            if parent:
                p = db.execute('SELECT * FROM guarded_grants WHERE id=?', (parent,)).fetchone()
                if p is None:
                    raise PermissionError('Unknown parent')
                self._grant(db, parent, task, tool, p['actor'])
                if expires > p['expires']:
                    raise PermissionError('Child outlives parent')
            db.execute('INSERT INTO guarded_grants(id,task,tool,actor,expires,parent) VALUES(?,?,?,?,?,?)',
                       (grant_id, task, tool, actor, expires, parent))
        return grant_id

    def revoke(self, *, task=None, grant_id=None):
        if (task is None) == (grant_id is None):
            raise ValueError('Specify exactly one revocation target')
        with self.transaction() as db:
            table, column, key = ('guarded_tasks', 'task', task) if task else ('guarded_grants', 'id', grant_id)
            if db.execute(f'UPDATE {table} SET revoked=1 WHERE {column}=?', (key,)).rowcount != 1:
                raise PermissionError('Unknown revocation target')
        # Dispatched/unknown effects still need reconciliation; never refund here.

    def propose_refinement(self, proposal: RefinementProposal) -> str:
        """Untrusted channel: record an agent-proposed authority change.

        This mints nothing. The row stays pending until a host review either
        applies it (with a host budget) or denies it. Duplicate proposals for
        an already-contracted tool and proposals for unconfigured, revoked or
        expired tasks fail closed here. An identical PENDING proposal returns
        its existing id instead of creating duplicate review noise.
        """
        identifier(proposal.task)
        if proposal.tool not in ADAPTERS:
            raise ValueError('Unknown adapter tool')
        if not isinstance(proposal.stage, str) or not proposal.stage or not isinstance(proposal.next_stage, str) or not proposal.next_stage:
            raise ValueError('Nonempty stage names required')
        if proposal.evidence_kind is not None and (not isinstance(proposal.evidence_kind, str) or not proposal.evidence_kind):
            raise ValueError('Evidence kind must be None or a nonempty string')
        if type(proposal.requested_calls) is not int or type(proposal.requested_units) is not int or min(proposal.requested_calls, proposal.requested_units) <= 0:
            raise ValueError('Requested budget must be positive integers')
        ADAPTERS[proposal.tool].validate(dict(proposal.bindings), dict(proposal.bindings))
        frozen = json.loads(self.canonical(proposal.bindings))
        proposal_id = str(uuid4())
        proposal_text = self.canonical({'stage': proposal.stage, 'next_stage': proposal.next_stage,
                                        'evidence_kind': proposal.evidence_kind,
                                        'requested_calls': proposal.requested_calls,
                                        'requested_units': proposal.requested_units,
                                        'bindings': frozen, 'note': str(proposal.note)[:500]})
        with self.transaction() as db:
            self._scope(db, proposal.task)
            if db.execute('SELECT 1 FROM contracts WHERE task=? AND tool=?', (proposal.task, proposal.tool)).fetchone():
                raise PermissionError('Tool already has an exact task contract')
            duplicate = db.execute("SELECT id FROM guarded_refinements WHERE task=? AND tool=? AND proposal=? AND state='pending'",
                                  (proposal.task, proposal.tool, proposal_text)).fetchone()
            if duplicate:
                return duplicate['id']
            db.execute('INSERT INTO guarded_refinements VALUES(?,?,?,?,?,?)',
                       (proposal_id, proposal.task, proposal.tool, proposal_text, 'pending', None))
        return proposal_id

    def review_refinement(self, proposal_id, *, reviewer, budget_calls, budget_units, value_verify=None):
        """HOST control plane: one human review per shape, then attested reuse.

        The reviewer runs outside the write transaction (a human may take
        unbounded time) and sees a deep copy; it can only return exactly True
        or veto. Budget authority stays host-only: the proposal's requested
        budget is advisory data and is never granted. An approved shape enters
        the template cache; reuse on another task skips the human only when
        value_verify (host code) attests that task's fresh values AND the
        reused stage mapping (the cached shape carries a stage transition
        whose safety can be task-specific, so the verifier receives the rule
        too and may veto it). Without a verifier, reuse falls back to the
        human; an explicit verifier rejection fails closed with no human
        fallback. value_verify and the reviewer both receive deep copies and
        cannot mutate what gets applied. The apply phase re-validates the
        pending state and task scope, then atomically mints the exact
        contract, the host budget and the transition rule.
        """
        if not callable(reviewer):
            raise ValueError('Reviewer callback required')
        if type(budget_calls) is not int or type(budget_units) is not int or min(budget_calls, budget_units) <= 0:
            raise ValueError('Host budget must be positive integers')

        def settle_denied(route):
            # rowcount==0 means another reviewer settled this concurrently;
            # never report a denial that did not actually happen.
            with self.transaction() as db:
                if db.execute("UPDATE guarded_refinements SET state='denied',route=?"
                              " WHERE id=? AND state='pending'", (route, proposal_id)).rowcount != 1:
                    raise PermissionError('Refinement proposal was settled concurrently')

        with self.transaction() as db:
            row = db.execute('SELECT * FROM guarded_refinements WHERE id=?', (proposal_id,)).fetchone()
            if row is None or row['state'] != 'pending':
                raise PermissionError('Unknown or non-pending refinement proposal')
            scope = self._scope(db, row['task'])
            spec = json.loads(row['proposal'])
            rule = {k: spec[k] for k in ('stage', 'next_stage', 'evidence_kind')}
            shape = self.canonical({'tool': row['tool'], 'revision': scope['revision'],
                                    'fields': sorted(ADAPTERS[row['tool']].fields), 'rule': rule})
            key = hashlib.sha256(shape.encode()).hexdigest()
            template_hit = bool(db.execute('SELECT 1 FROM guarded_refinement_templates WHERE key=? AND shape=?',
                                           (key, shape)).fetchone())
        route = None
        if template_hit:
            if value_verify is not None:
                if not value_verify(row['task'], row['tool'],
                                    json.loads(self.canonical(spec['bindings'])),
                                    json.loads(self.canonical(rule))):
                    settle_denied('value_verification_failed')
                    return {'proposal_id': proposal_id, 'state': 'denied',
                            'route': 'value_verification_failed'}
                route = 'template_attested'
        if route is None:
            view = json.loads(self.canonical({'task': row['task'], 'tool': row['tool'],
                                               **{k: spec[k] for k in ('stage', 'next_stage', 'evidence_kind',
                                                                       'requested_calls', 'requested_units',
                                                                       'bindings', 'note')}}))
            try:
                approved = reviewer(view)
            except Exception:
                settle_denied('review_error')
                return {'proposal_id': proposal_id, 'state': 'denied', 'route': 'review_error'}
            if approved is not True:
                settle_denied('review_denied')
                return {'proposal_id': proposal_id, 'state': 'denied', 'route': 'review_denied'}
            route = 'template_reviewed' if template_hit else 'slow'
        with self.transaction() as db:
            current = db.execute('SELECT * FROM guarded_refinements WHERE id=?', (proposal_id,)).fetchone()
            if current is None or current['state'] != 'pending' or current['proposal'] != row['proposal']:
                raise PermissionError('Refinement proposal changed or already settled')
            self._scope(db, row['task'])
            if db.execute('SELECT 1 FROM contracts WHERE task=? AND tool=?', (row['task'], row['tool'])).fetchone():
                raise PermissionError('Tool already has an exact task contract')
            if route == 'slow':
                db.execute('INSERT OR IGNORE INTO guarded_refinement_templates VALUES(?,?)', (key, shape))
            # The minted rule also flows into the activation template cache
            # (guarded_templates); that is safe because every dispatch still
            # requires full equality with these attested exact bindings.
            budget_id = str(uuid4())
            ADAPTERS[row['tool']].validate(spec['bindings'], spec['bindings'])
            db.execute('INSERT INTO budgets VALUES(?,?,?,?)', (budget_id, row['task'], budget_calls, budget_units))
            db.execute('INSERT INTO contracts VALUES(?,?,?,?)',
                       (row['task'], row['tool'], budget_id, self.canonical(spec['bindings'])))
            db.execute('INSERT INTO guarded_rules VALUES(?,?,?)',
                       (row['task'], row['tool'], self.canonical(rule)))
            db.execute("UPDATE guarded_refinements SET state='applied',route=? WHERE id=?", (route, proposal_id))
        return {'proposal_id': proposal_id, 'state': 'applied', 'route': route,
                'granted_calls': budget_calls, 'granted_units': budget_units,
                'requested_calls': spec['requested_calls'], 'requested_units': spec['requested_units']}

    def refinements(self, task=None):
        with self.transaction() as db:
            rows = (db.execute('SELECT * FROM guarded_refinements WHERE task=?', (task,)) if task
                    else db.execute('SELECT * FROM guarded_refinements')).fetchall()
        return [dict(r) for r in rows]

    def _shape(self, db, task, tool):
        scope = self._scope(db, task)
        row = db.execute('SELECT rule FROM guarded_rules WHERE task=? AND tool=?', (task, tool)).fetchone()
        if row is None:
            raise PermissionError('No host transition rule')
        return self.canonical({'tool': tool, 'revision': scope['revision'],
                               'fields': sorted(ADAPTERS[tool].fields), 'rule': json.loads(row['rule'])})

    def template_key(self, task, tool, *, remember=False):
        """Cache rule shape only; task targets and budgets never enter the cache."""
        with self.transaction() as db:
            shape = self._shape(db, task, tool)
            key = hashlib.sha256(shape.encode()).hexdigest()
            if remember:
                db.execute('INSERT OR IGNORE INTO guarded_templates VALUES(?,?)', (key, shape))
            hit = db.execute('SELECT 1 FROM guarded_templates WHERE key=? AND shape=?', (key, shape)).fetchone()
            return key, bool(hit)

    def check_action(self, task, tool, arguments):
        """Validate scope and exact host authority BEFORE asking a model reviewer."""
        with self.transaction() as db:
            scope = self._scope(db, task)
            row = db.execute('SELECT bindings FROM contracts WHERE task=? AND tool=?', (task, tool)).fetchone()
            if row is None:
                raise PermissionError('Missing authenticated contract')
            ADAPTERS[tool].validate(json.loads(row['bindings']), arguments)
            rule_row = db.execute('SELECT rule FROM guarded_rules WHERE task=? AND tool=?', (task, tool)).fetchone()
            if rule_row is None:
                raise PermissionError('Missing host transition rule')
            rule = json.loads(rule_row['rule'])
            if scope['holder'] or scope['stage'] != rule['stage']:
                raise PermissionError('Workflow token unavailable or wrong stage')
            encoded = self.canonical(arguments)
            for kind in {rule['evidence_kind'], ADAPTERS[tool].evidence_kind} - {None}:
                if not db.execute('SELECT 1 FROM evidence WHERE task=? AND kind=? AND payload=?', (task, kind, encoded)).fetchone():
                    raise PermissionError('Missing exact transition evidence')
            budget = db.execute('SELECT b.* FROM budgets b JOIN contracts c ON b.id=c.budget WHERE c.task=? AND c.tool=?', (task, tool)).fetchone()
            cost = ADAPTERS[tool].validate(json.loads(row['bindings']), arguments)
            if budget['calls'] < 1 or budget['units'] < cost:
                raise PermissionError('Budget exhausted or already reserved')
            return dict(scope)

    def _reserve_guard(self, db, contract, arguments, operation, grant_id, actor):
        scope = self._scope(db, contract['task'])
        self._grant(db, grant_id, contract['task'], contract['tool'], actor) if grant_id else None
        if not grant_id:
            raise PermissionError('Active task grant required')
        rule = json.loads(db.execute('SELECT rule FROM guarded_rules WHERE task=? AND tool=?', (contract['task'], contract['tool'])).fetchone()['rule'])
        if scope['holder'] or scope['stage'] != rule['stage']:
            raise PermissionError('Workflow token unavailable or wrong stage')
        if rule['evidence_kind'] and not db.execute('SELECT 1 FROM evidence WHERE task=? AND kind=? AND payload=?',
                                                   (contract['task'], rule['evidence_kind'], self.canonical(arguments))).fetchone():
            raise PermissionError('Missing exact transition evidence')
        db.execute('UPDATE guarded_tasks SET holder=? WHERE task=?', (operation, contract['task']))
        db.execute('INSERT INTO guarded_reservations VALUES(?,?,?,?)', (operation, grant_id, actor, rule['next_stage']))

    def _settle_guard(self, db, operation, target):
        row = db.execute('SELECT * FROM guarded_reservations WHERE operation=?', (operation['id'],)).fetchone()
        if row is None:
            raise PermissionError('Missing guarded reservation')
        if target == 'dispatched':
            self._scope(db, operation['task'])
            self._grant(db, row['grant_id'], operation['task'], operation['tool'], row['actor'])
        if target in ('committed', 'aborted'):
            if target == 'committed':
                db.execute('UPDATE guarded_tasks SET stage=?,holder=NULL WHERE task=? AND holder=?',
                           (row['next_stage'], operation['task'], operation['id']))
            else:
                db.execute('UPDATE guarded_tasks SET holder=NULL WHERE task=? AND holder=?', (operation['task'], operation['id']))

    def marking(self, task):
        with self.transaction() as db:
            scope = db.execute('SELECT * FROM guarded_tasks WHERE task=?', (task,)).fetchone()
            return {'workflow': dict(scope) if scope else None,
                    'budgets': [dict(r) for r in db.execute('SELECT * FROM budgets WHERE task=?', (task,))],
                    'operations': [dict(r) for r in db.execute('SELECT * FROM operations WHERE task=?', (task,))]}


@dataclass(frozen=True)
class ActivationPolicy:
    """Pinned HOST effect metadata. Never derive from untrusted descriptions."""
    effect: str = 'unknown'
    external_disclosure: bool = True
    mutates_state: bool = True
    revision: str = ''

    @property
    def automatic(self):
        return (bool(self.revision) and self.effect in {'bounded_local_read', 'local_compute'}
                and not self.external_disclosure and not self.mutates_state)


class GuardedBoundary(TransactionBoundary):
    """Fast-path grants, bounded slow-path review, then the SAME atomic executor.

    actor/reviewer come from host code. The reviewer may veto or approve activation
    of a preauthorized exact contract. Its approval never enlarges that contract.
    """
    def __init__(self, inner, net, bindings, read_guard=None, *, actor, reviewer, activation_policies=None, recovery_advice=None, refine_missing=False):
        if not isinstance(net, GuardedTransactionNet):
            raise TypeError('GuardedTransactionNet required')
        super().__init__(inner, net, bindings, read_guard)
        self.actor, self.reviewer = actor, reviewer
        self.active = {}
        self.activation_policies = dict(activation_policies or {})
        self.recovery_advice = recovery_advice
        self.refine_missing = refine_missing
        self.proposed_refinements = set()

    def run_function(self, env, function, kwargs, raise_on_error=False):
        result = super().run_function(env, function, kwargs, raise_on_error)
        value, error = result
        if error and self.refine_missing:
            error = self._propose_missing_refinement(function, kwargs, error)
        if error and self.recovery_advice is not None:
            # Advice is data for a NEW proposal; it neither executes nor grants.
            advice = self.recovery_advice(function)
            if advice:
                error += '\nHost recovery guidance: ' + json.dumps(advice, sort_keys=True)
        return value, error

    def _propose_missing_refinement(self, function, kwargs, error):
        """Record one pending refinement proposal per distinct missing call.

        The denied call itself is the LLM proposal: it grants nothing until a
        host review approves its shape. Duplicate denials do not spam
        duplicate proposals; every other denial reason is left untouched.
        """
        binding = self.bindings.get(function)
        if binding is None or not any(message in error for message in
                                      ('Missing authenticated contract', 'Missing host transition rule')):
            return error
        try:
            arguments = json.loads(self.net.canonical(binding.encode(json.loads(self.net.canonical(kwargs)))))
            stage = (self.net.marking(binding.task_id)['workflow'] or {}).get('stage')
        except Exception:
            return error
        if not stage:
            return error
        key = (binding.task_id, binding.adapter, self.net.canonical(arguments))
        if key in self.proposed_refinements:
            return error
        try:
            proposal_id = self.net.propose_refinement(RefinementProposal(
                task=binding.task_id, tool=binding.adapter, bindings=arguments,
                stage=stage, next_stage=stage, note='agent proposal: missing host authority'))
        except (ValueError, PermissionError):
            return error
        self.proposed_refinements.add(key)
        self.decisions.append({'tool': function, 'state': 'denied', 'refinement_proposed': proposal_id})
        return error + '\nRefinement proposal pending host review: ' + proposal_id

    def _authorization(self, binding, canonical, raw_arguments=None):
        scope = self.net.check_action(binding.task_id, binding.adapter, canonical)
        key = (binding.task_id, binding.adapter)
        grant_id = self.active.get(key)
        route = 'fast'
        if grant_id is None:
            policy = self.activation_policies.get(key, ActivationPolicy())
            auto = policy.automatic and binding.adapter == 'benchmark.sandbox.invoke'
            # Only the explicitly inert benchmark adapter currently has a
            # reviewed local-read integration. Payment/send/verify never auto-pass.
            _, hit = self.net.template_key(*key)
            # Explicit high-risk review is never bypassed by another task's cache.
            # Legacy callers retain their original cache behavior.
            if self.activation_policies and not auto:
                hit = False
            if not hit and not auto:
                # A deep copy prevents reviewer mutation of reserved arguments.
                proposal = json.loads(self.net.canonical({'task': key[0], 'tool': key[1], 'arguments': canonical, 'raw_arguments': raw_arguments,
                    'stage': scope['stage'], 'host_contract_verified': True}))
                try:
                    approved = self.reviewer(proposal)
                except Exception as exc:
                    raise PermissionError('Slow-path review failed: ' + type(exc).__name__) from exc
                if approved is not True:
                    raise PermissionError('Slow-path review denied activation')
                self.net.template_key(*key, remember=True)
            route = 'bounded_read_auto' if auto else ('template' if hit else 'slow')
            grant_id = self.net.activate(*key, actor=self.actor, expires=scope['expires'])
            self.active[key] = grant_id
        self.decisions.append({'route': route, 'task': key[0], 'grant': grant_id})
        return {'grant_id': grant_id, 'actor': self.actor}
