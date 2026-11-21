from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from toolfence.guarded_net import GuardedTransactionNet, GuardedBoundary, TransitionRule, EvidenceStep
from toolfence.transaction_net import TaskContract, ObjectRef, Receipt, HostToolBinding


def uid():
    return str(uuid4())


def setup(tmp_path, *, task=None, account='alice', initial='ready', steps=None, clock=lambda: 100):
    net = GuardedTransactionNet(str(tmp_path / 'net.sqlite'), clock=clock)
    task = task or uid()
    args = {'account_ref': ObjectRef('bank', 'account', account, '1').encode(), 'amount_minor': 100, 'currency': 'USD'}
    net.issue(TaskContract(task, uid(), 'payment.transfer', args, max_units=100))
    net.configure(task, initial_stage=initial, rules={'payment.transfer': TransitionRule('ready', 'done')},
                  steps=steps, revision='trusted-adapter-v1', expires=200)
    return net, task, args


def grant(net, task, actor='A', **kwargs):
    return net.activate(task, 'payment.transfer', actor=actor, expires=190, **kwargs)


def reserve(net, task, args, g, actor='A'):
    op = uid()
    net.reserve(op, task, 'payment.transfer', args, grant_id=g, actor=actor)
    return op


def test_workflow_evidence_and_effect_share_marking(tmp_path):
    payload = {'order': 'O17', 'product': 'P17'}
    step = EvidenceStep('cart_ready', 'ready', 'order_bound', payload)
    net, task, args = setup(tmp_path, initial='cart_ready', steps={'order': step})
    g = grant(net, task)
    with pytest.raises(PermissionError):
        reserve(net, task, args, g)
    net.ingest(task, 'order_bound', {'order': 'O99', 'product': 'P17'}, lambda *_: True)
    with pytest.raises(PermissionError):
        net.advance(task, 'order')
    net.ingest(task, 'order_bound', payload, lambda *_: True)
    net.advance(task, 'order')
    op = reserve(net, task, args, g)
    assert net.marking(task)['workflow']['stage'] == 'ready'
    assert net.marking(task)['workflow']['holder'] == op
    net.dispatch(op, lambda op, *_: Receipt(op, 'committed'))
    assert net.marking(task)['workflow']['stage'] == 'done'
    assert net.marking(task)['workflow']['holder'] is None


def test_delegates_share_budget_across_connections(tmp_path):
    net, task, args = setup(tmp_path)
    parent = grant(net, task)
    children = [grant(net, task, str(i), parent=parent) for i in range(12)]
    def run(i):
        other = GuardedTransactionNet(net.path, clock=lambda: 100)
        try:
            return reserve(other, task, args, children[i], str(i))
        except PermissionError:
            return None
    with ThreadPoolExecutor(max_workers=6) as pool:
        winners = list(pool.map(run, range(12)))
    assert sum(w is not None for w in winners) == 1
    assert net.marking(task)['budgets'][0]['calls'] == 0


@pytest.mark.parametrize('target', ['task', 'grant'])
def test_revoke_between_reserve_and_dispatch(tmp_path, target):
    net, task, args = setup(tmp_path)
    g = grant(net, task)
    op = reserve(net, task, args, g)
    net.revoke(**({'task': task} if target == 'task' else {'grant_id': g}))
    with pytest.raises(PermissionError):
        net.dispatch(op, lambda *_: pytest.fail('must not execute'))
    net.abort(op)
    assert net.marking(task)['budgets'][0]['calls'] == 1
    with pytest.raises(PermissionError):
        reserve(net, task, args, g)


def test_parent_revocation_and_actor_binding(tmp_path):
    net, task, args = setup(tmp_path)
    parent = grant(net, task)
    child = grant(net, task, 'B', parent=parent)
    with pytest.raises(PermissionError):
        reserve(net, task, args, child, 'A')
    net.revoke(grant_id=parent)
    with pytest.raises(PermissionError):
        reserve(net, task, args, child, 'B')


def test_unknown_survives_restart_and_revocation(tmp_path):
    net, task, args = setup(tmp_path)
    op = reserve(net, task, args, grant(net, task))
    net.dispatch(op, lambda op, *_: Receipt(op, 'unknown'))
    net = GuardedTransactionNet(net.path, clock=lambda: 100)
    net.revoke(task=task)
    assert net.marking(task)['workflow']['holder'] == op
    net.reconcile(op, lambda op: Receipt(op, 'committed'))
    assert net.marking(task)['workflow']['stage'] == 'done'
    assert net.marking(task)['budgets'][0]['calls'] == 0


def test_not_applied_releases_stage_and_budget(tmp_path):
    net, task, args = setup(tmp_path)
    g = grant(net, task)
    op = reserve(net, task, args, g)
    net.dispatch(op, lambda op, *_: Receipt(op, 'not_applied'))
    assert net.marking(task)['workflow']['stage'] == 'ready'
    assert net.marking(task)['budgets'][0]['calls'] == 1
    reserve(net, task, args, g)


def test_expiry_rechecked_at_dispatch_and_child_cannot_extend(tmp_path):
    now = [100]
    net, task, args = setup(tmp_path, clock=lambda: now[0])
    g = grant(net, task)
    with pytest.raises(PermissionError):
        net.activate(task, 'payment.transfer', actor='B', expires=195, parent=g)
    op = reserve(net, task, args, g)
    now[0] = 191
    with pytest.raises(PermissionError):
        net.dispatch(op, lambda *_: pytest.fail('expired'))
    net.abort(op)


def boundary(net, task, args, reviews, *, reviewer=None):
    class Host:
        def __init__(self):
            self.executed = []
        def run_function(self, env, function, kwargs, **_):
            self.executed.append(dict(kwargs))
            return 'done', None
    host = Host()
    binding = HostToolBinding(task, 'payment.transfer', lambda a: a,
                              lambda op, result: Receipt(op, 'committed'), frozenset(args))
    def review(proposal):
        reviews.append(proposal)
        return True
    return GuardedBoundary(host, net, {'pay': binding}, actor='A', reviewer=reviewer or review), host


def test_slow_path_template_reuse_needs_current_task_authority(tmp_path):
    net, task, args = setup(tmp_path)
    reviews = []
    guard, host = boundary(net, task, args, reviews)
    assert guard.run_function(None, 'pay', {**args, 'amount_minor': 200})[1]
    assert reviews == [] and host.executed == []
    assert guard.run_function(None, 'pay', args)[1] is None
    assert len(reviews) == 1
    assert guard.run_function(None, 'pay', args)[1]
    assert len(host.executed) == 1
    net2, task2, args2 = setup(tmp_path, account='bob')
    guard2, host2 = boundary(net2, task2, args2, reviews)
    assert guard2.run_function(None, 'pay', args)[1]  # prior task's values
    assert guard2.run_function(None, 'pay', args2)[1] is None
    assert len(reviews) == 1
    assert host2.executed == [args2]
    assert any(d.get('route') == 'template' for d in guard2.decisions)
    with pytest.raises(PermissionError):
        net.template_key(uid(), 'payment.transfer')


def test_review_deny_and_mutation_cannot_authorize(tmp_path):
    net, task, args = setup(tmp_path)
    guard, host = boundary(net, task, args, [], reviewer=lambda p: False)
    assert guard.run_function(None, 'pay', args)[1]
    assert host.executed == []
    def mutate(p):
        p['arguments']['amount_minor'] = 999
        return True
    guard, host = boundary(net, task, args, [], reviewer=mutate)
    assert guard.run_function(None, 'pay', args)[1] is None
    assert host.executed == [args]


def test_guarded_net_cannot_execute_unconfigured_or_without_grant(tmp_path):
    net, task, args = setup(tmp_path)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'payment.transfer', args)
    assert net.marking(task)['budgets'][0]['calls'] == 1
    wrong = grant(net, task)
    net2, task2, args2 = setup(tmp_path, account='bob')
    with pytest.raises(PermissionError):
        reserve(net2, task2, args2, wrong)


def test_dispatch_revocation_boundary_reports_denied_and_releases(tmp_path):
    net, task, args = setup(tmp_path)
    original_dispatch = net.dispatch
    def revoke_then_dispatch(op, transport):
        net.revoke(task=task)
        return original_dispatch(op, transport)
    net.dispatch = revoke_then_dispatch
    guard, host = boundary(net, task, args, [])
    _, error = guard.run_function(None, 'pay', args)
    assert 'revoked' in error
    assert host.executed == []
    assert guard.decisions[-1]['state'] == 'denied'
    assert net.marking(task)['budgets'][0]['calls'] == 1
    assert net.marking(task)['workflow']['holder'] is None


def test_fast_path_after_known_no_effect_does_not_regrant(tmp_path):
    net, task, args = setup(tmp_path)
    reviews, attempts = [], []
    class Host:
        def run_function(self, *_, **__):
            attempts.append(1)
            return 'not_applied' if len(attempts) == 1 else 'committed', None
    binding = HostToolBinding(task, 'payment.transfer', lambda a: a,
                              lambda op, result: Receipt(op, result[0]), frozenset(args))
    def review(proposal):
        reviews.append(proposal)
        return True
    guard = GuardedBoundary(Host(), net, {'pay': binding}, actor='A', reviewer=review)
    guard.run_function(None, 'pay', args)
    assert net.marking(task)['budgets'][0]['calls'] == 1
    guard.run_function(None, 'pay', args)
    assert len(reviews) == 1
    assert net.marking(task)['budgets'][0]['calls'] == 0
    assert any(d.get('route') == 'fast' for d in guard.decisions)


def test_order_workflow_example(tmp_path):
    from experiments.run_guarded_net_cases import run
    result = run(tmp_path / 'example')
    assert [case['allowed'] for case in result['trace']] == [False, False, False, True, False]
    assert result['trace'][-1]['marking']['workflow']['stage'] == 'completed'


def test_review_timeout_cannot_cache_or_dispatch(tmp_path):
    net, task, args = setup(tmp_path)
    def timeout(_):
        raise TimeoutError('model unavailable')
    guard, host = boundary(net, task, args, [], reviewer=timeout)
    _, error = guard.run_function(None, 'pay', args)
    assert 'Slow-path review failed' in error
    assert host.executed == []
    assert not net.template_key(task, 'payment.transfer')[1]
    assert net.marking(task)['budgets'][0]['calls'] == 1


def test_existing_hybrid_runtime_dispatches_through_unified_net(tmp_path):
    from toolfence.colored_runtime import TransactionalColoredToolFenceRuntime
    from toolfence.runtime import ToolFenceJudge
    from tests.unit.test_toolfence import Backend, Runtime, monitor_for
    net = GuardedTransactionNet(str(tmp_path / 'hybrid.sqlite'), clock=lambda: 100)
    task = uid()
    iban = 'GB29NWBK60161331926819'
    raw = {'recipient': iban, 'amount': 4.0, 'subject': 'Refund', 'date': '2022-04-01'}
    canonical = {'account_ref': ObjectRef('bank', 'account', iban, '1').encode(), 'amount_minor': 400, 'currency': 'GBP'}
    net.issue(TaskContract(task, uid(), 'payment.transfer', canonical, max_units=400))
    net.configure(task, initial_stage='ready', rules={'payment.transfer': TransitionRule('ready', 'done')}, revision='bank-v1', expires=200)
    def encode(args):
        if args != raw:
            raise PermissionError('Host bindings mismatch')
        return canonical
    binding = HostToolBinding(task, 'payment.transfer', encode, lambda op, _: Receipt(op, 'committed'), frozenset(raw))
    inner = Runtime(({'message': 'sent'}, None))
    runtime = TransactionalColoredToolFenceRuntime(
        inner, monitor_for('send_money', effect='financial'),
        ToolFenceJudge(Backend('{"decision":"allow","reason":"user requested transfer"}')),
        query='Send 4 pounds to ' + iban, tools=[], transaction_net=net,
        host_bindings={'send_money': binding}, host_actor='A', host_reviewer=lambda _: True)
    assert runtime.run_function(None, 'send_money', raw)[1] is None
    assert net.marking(task)['workflow']['stage'] == 'done'
    assert len(inner.calls) == 1
    assert runtime.run_function(None, 'send_money', raw)[1]
    assert len(inner.calls) == 1
