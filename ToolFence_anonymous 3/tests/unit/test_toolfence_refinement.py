"""Unit coverage for the reviewed refinement channel.

The refinement channel deliberately widens the activation principle: model
output may PROPOSE new exact contracts/rules, a host reviewer approves each
shape once, budgets remain host-only, and template reuse requires host value
attestation. These are mechanism checks with zero model calls.
"""
from uuid import uuid4

import pytest

from toolfence.guarded_net import GuardedTransactionNet, GuardedBoundary, TransitionRule, RefinementProposal
from toolfence.transaction_net import TaskContract, ObjectRef, Receipt, HostToolBinding


def uid():
    return str(uuid4())


def configured(tmp_path, *, account='alice', revision='trusted-adapter-v1'):
    net = GuardedTransactionNet(str(tmp_path / 'net.sqlite'), clock=lambda: 100)
    task = uid()
    args = {'account_ref': ObjectRef('bank', 'account', account, '1').encode(), 'amount_minor': 100, 'currency': 'USD'}
    net.issue(TaskContract(task, uid(), 'payment.transfer', args, max_units=100))
    net.configure(task, initial_stage='ready', rules={'payment.transfer': TransitionRule('ready', 'done')},
                  revision=revision, expires=200)
    return net, task, args


def send_bindings(document='D3', recipient='user3@recipient.example.org'):
    return {'document_ref': ObjectRef('staged-host', 'document', document, '1').encode(),
            'recipient_ref': ObjectRef('staged-host', 'recipient', recipient, '1').encode(),
            'content_digest': 'a' * 64}


def propose(net, task, bindings=None, **kw):
    fields = dict(stage='ready', next_stage='done', requested_calls=5, requested_units=500, note='unit')
    fields.update(kw)
    return net.propose_refinement(RefinementProposal(task, 'document.send', bindings or send_bindings(), **fields))


def review_all(net, *pids):
    calls = []

    def reviewer(view):
        calls.append(view)
        return True

    results = [net.review_refinement(p, reviewer=reviewer, budget_calls=1, budget_units=1) for p in pids]
    return calls, results


def test_proposal_records_pending_but_mints_nothing(tmp_path):
    net, task, _ = configured(tmp_path)
    pid = propose(net, task)
    rows = net.refinements(task)
    assert len(rows) == 1 and rows[0]['state'] == 'pending' and rows[0]['route'] is None
    # Only the payment budget exists; a pending proposal minted no authority.
    assert len(net.marking(task)['budgets']) == 1


def test_review_denial_is_returned_and_never_cached(tmp_path):
    net, task, _ = configured(tmp_path)
    pid = propose(net, task)
    calls = []
    result = net.review_refinement(pid, reviewer=lambda p: calls.append(p) or False,
                                   budget_calls=1, budget_units=1)
    assert result['state'] == 'denied' and result['route'] == 'review_denied'
    assert net.refinements(task)[0]['state'] == 'denied'
    # A denial must not pollute the shape cache: the next identical shape is
    # reviewed again by a human instead of silently reusing a template.
    pid2 = propose(net, task)
    net.review_refinement(pid2, reviewer=lambda p: calls.append(p) or False, budget_calls=1, budget_units=1)
    assert len(calls) == 2


def test_reviewer_exception_fails_closed_without_caching(tmp_path):
    net, task, _ = configured(tmp_path)

    def explode(_):
        raise TimeoutError('reviewer unavailable')

    result = net.review_refinement(propose(net, task), reviewer=explode, budget_calls=1, budget_units=1)
    assert result['state'] == 'denied' and result['route'] == 'review_error'
    assert net.refinements(task)[0]['state'] == 'denied'


def test_approval_mints_host_budget_and_executes_once(tmp_path):
    net, task, _ = configured(tmp_path)
    args = send_bindings()
    pid = propose(net, task)
    result = net.review_refinement(pid, reviewer=lambda _: True, budget_calls=1, budget_units=1)
    assert result['state'] == 'applied' and result['route'] == 'slow'
    # The proposal requested 5 calls / 500 units; only the host numbers exist.
    assert result['granted_calls'] == 1 and result['requested_calls'] == 5
    grant = net.activate(task, 'document.send', actor='A', expires=190)
    net.reserve(uid(), task, 'document.send', args, grant_id=grant, actor='A')
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'document.send', args, grant_id=grant, actor='A')
    budgets = {b['calls'] for b in net.marking(task)['budgets']}
    assert 0 in budgets and 1 in budgets  # payment untouched, refined call spent


def test_reviewer_cannot_mutate_proposal_values(tmp_path):
    net, task, _ = configured(tmp_path)
    args = send_bindings()
    evil = {**args, 'recipient_ref': ObjectRef('staged-host', 'recipient', 'evil@x.example.org', '1').encode()}

    def mutate(view):
        view['bindings']['recipient_ref'] = evil['recipient_ref']
        return True

    net.review_refinement(propose(net, task), reviewer=mutate, budget_calls=1, budget_units=1)
    grant = net.activate(task, 'document.send', actor='A', expires=190)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'document.send', evil, grant_id=grant, actor='A')
    # The minted contract kept the ORIGINAL proposal values.
    net.reserve(uid(), task, 'document.send', args, grant_id=grant, actor='A')


def test_template_reuse_skips_reviewer_with_value_attestation(tmp_path):
    net, task1, _ = configured(tmp_path)
    calls, _ = review_all(net, propose(net, task1))
    assert len(calls) == 1
    _, task2, _ = configured(tmp_path, account='bob')
    pid2 = propose(net, task2, send_bindings(document='D9', recipient='user9@recipient.example.org'))
    result = net.review_refinement(pid2, reviewer=lambda p: calls.append(p) or True,
                                   budget_calls=1, budget_units=1, value_verify=lambda t, _, b, r: t == task2)
    assert result['state'] == 'applied' and result['route'] == 'template_attested'
    assert len(calls) == 1  # the human reviewed only the first shape


def test_template_reuse_without_verifier_falls_back_to_reviewer(tmp_path):
    net, task1, _ = configured(tmp_path)
    calls, _ = review_all(net, propose(net, task1))
    _, task2, _ = configured(tmp_path, account='bob')
    result = net.review_refinement(propose(net, task2), reviewer=lambda p: calls.append(p) or True,
                                   budget_calls=1, budget_units=1)
    assert result['state'] == 'applied' and result['route'] == 'template_reviewed'
    assert len(calls) == 2


def test_failed_value_verification_denies_without_human_fallback(tmp_path):
    net, task1, _ = configured(tmp_path)
    calls, _ = review_all(net, propose(net, task1))
    _, task2, _ = configured(tmp_path, account='bob')
    result = net.review_refinement(propose(net, task2), reviewer=lambda p: calls.append(p) or True,
                                   budget_calls=1, budget_units=1, value_verify=lambda *_: False)
    assert result['state'] == 'denied' and result['route'] == 'value_verification_failed'
    assert len(calls) == 1  # the explicit host verifier failed closed; no human


def test_revision_change_never_reuses_templates(tmp_path):
    net, task1, _ = configured(tmp_path)
    calls, _ = review_all(net, propose(net, task1))
    _, task2, _ = configured(tmp_path, account='bob', revision='trusted-adapter-v2')
    result = net.review_refinement(propose(net, task2), reviewer=lambda p: calls.append(p) or True,
                                   budget_calls=1, budget_units=1, value_verify=lambda *_: True)
    assert result['route'] == 'slow' and len(calls) == 2


def test_revoked_task_cannot_apply_refinement(tmp_path):
    net, task, _ = configured(tmp_path)
    pid = propose(net, task)
    net.revoke(task=task)
    with pytest.raises(PermissionError):
        net.review_refinement(pid, reviewer=lambda _: True, budget_calls=1, budget_units=1)
    # The scope failure settled nothing; the proposal is still pending.
    assert net.refinements(task)[0]['state'] == 'pending'


def test_expired_task_cannot_propose(tmp_path):
    now = [100]
    net = GuardedTransactionNet(str(tmp_path / 'expired.sqlite'), clock=lambda: now[0])
    task = uid()
    net.issue(TaskContract(task, uid(), 'payment.transfer',
                            {'account_ref': ObjectRef('bank', 'account', 'alice', '1').encode(),
                             'amount_minor': 100, 'currency': 'USD'}))
    net.configure(task, initial_stage='ready',
                  rules={'payment.transfer': TransitionRule('ready', 'done')},
                  revision='trusted-adapter-v1', expires=200)
    now[0] = 201
    with pytest.raises(PermissionError):
        propose(net, task)


def test_host_verifier_receives_and_can_veto_the_reused_rule(tmp_path):
    net, task1, _ = configured(tmp_path)
    review_all(net, propose(net, task1))
    _, task2, _ = configured(tmp_path, account='bob')
    rules = []

    def verify(task, tool, bindings, rule):
        rules.append(rule)
        return False  # host rejects the cached stage mapping for this task

    result = net.review_refinement(propose(net, task2),
                                   reviewer=lambda p: pytest.fail('human must not be consulted'),
                                   budget_calls=1, budget_units=1, value_verify=verify)
    assert result['state'] == 'denied' and result['route'] == 'value_verification_failed'
    assert rules == [{'stage': 'ready', 'next_stage': 'done', 'evidence_kind': None}]


def test_value_verifier_cannot_mutate_applied_values(tmp_path):
    net, task1, _ = configured(tmp_path)
    review_all(net, propose(net, task1))
    _, task2, _ = configured(tmp_path, account='bob')
    args = send_bindings(document='D9', recipient='user9@recipient.example.org')
    evil = ObjectRef('staged-host', 'recipient', 'evil@x.example.org', '1').encode()

    def verify(task, tool, bindings, rule):
        bindings['recipient_ref'] = evil
        return True

    result = net.review_refinement(propose(net, task2, args), reviewer=lambda _: True,
                                   budget_calls=1, budget_units=1, value_verify=verify)
    assert result['state'] == 'applied'
    grant = net.activate(task2, 'document.send', actor='A', expires=190)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task2, 'document.send', {**args, 'recipient_ref': evil}, grant_id=grant, actor='A')
    net.reserve(uid(), task2, 'document.send', args, grant_id=grant, actor='A')


def test_identical_pending_proposal_is_deduplicated(tmp_path):
    net, task, _ = configured(tmp_path)
    first = propose(net, task)
    assert propose(net, task) == first
    assert len(net.refinements(task)) == 1


def test_settled_proposal_cannot_be_reviewed_twice(tmp_path):
    net, task1, _ = configured(tmp_path)
    applied = propose(net, task1)
    net.review_refinement(applied, reviewer=lambda _: True, budget_calls=1, budget_units=1)
    with pytest.raises(PermissionError):
        net.review_refinement(applied, reviewer=lambda _: True, budget_calls=1, budget_units=1)
    _, task2, _ = configured(tmp_path, account='bob')
    denied = propose(net, task2)
    net.review_refinement(denied, reviewer=lambda _: False, budget_calls=1, budget_units=1)
    with pytest.raises(PermissionError):
        net.review_refinement(denied, reviewer=lambda _: True, budget_calls=1, budget_units=1)


def test_propose_rejects_contracted_tools_and_unconfigured_tasks(tmp_path):
    net, task, args = configured(tmp_path)
    with pytest.raises(PermissionError):
        net.propose_refinement(RefinementProposal(task, 'payment.transfer', args, 'ready', 'done'))
    with pytest.raises(PermissionError):
        net.propose_refinement(RefinementProposal(uid(), 'document.send', send_bindings(), 'ready', 'done'))
    with pytest.raises(PermissionError):
        net.propose_refinement(RefinementProposal(task, 'document.send', {'document_ref': 'x'}, 'ready', 'done'))


def boundary(net, task, send_args, reviews, *, refine_missing):
    class Host:
        def __init__(self):
            self.executed = []

        def run_function(self, env, function, kwargs, **_):
            self.executed.append((function, dict(kwargs)))
            return 'done', None

    host = Host()
    binding = HostToolBinding(task, 'document.send', lambda a: a,
                              lambda op, result: Receipt(op, 'committed'), frozenset(send_args))

    def review(proposal):
        reviews.append(proposal)
        return True

    return GuardedBoundary(host, net, {'send_document': binding}, actor='A', reviewer=review,
                           refine_missing=refine_missing), host


def test_boundary_records_refinement_for_missing_authority(tmp_path):
    net, task, _ = configured(tmp_path)
    args = send_bindings()
    reviews = []
    guard, host = boundary(net, task, args, reviews, refine_missing=True)
    assert guard.run_function(None, 'send_document', args)[1]
    assert host.executed == []
    pending = [r for r in net.refinements(task) if r['state'] == 'pending']
    assert len(pending) == 1
    assert any('refinement_proposed' in d for d in guard.decisions)
    # A duplicate denial does not spam duplicate proposals.
    guard.run_function(None, 'send_document', args)
    assert len(net.refinements(task)) == 1
    result = net.review_refinement(pending[0]['id'], reviewer=lambda _: True, budget_calls=1, budget_units=1)
    assert result['state'] == 'applied'
    assert guard.run_function(None, 'send_document', args)[1] is None
    assert host.executed == [('send_document', args)]


def test_boundary_default_records_no_refinement(tmp_path):
    net, task, _ = configured(tmp_path)
    args = send_bindings()
    guard, host = boundary(net, task, args, [], refine_missing=False)
    assert guard.run_function(None, 'send_document', args)[1]
    assert host.executed == []
    assert net.refinements(task) == []
    assert not any('refinement_proposed' in d for d in guard.decisions)


def test_scripted_refinement_case_suite(tmp_path):
    from experiments.run_refinement_cases import run
    result = run(tmp_path / 'refinement')
    states = [r['state'] for r in result['refinements']]
    assert states.count('applied') == 2 and states.count('denied') == 3
    # Exactly three human refinement reviews: deny, approve, fallback-deny.
    assert result['refinement_reviews'] == 3 and result['human_reviews'] >= 3
