"""Reproducible refinement cases: agent proposes, host reviews once, reuse is attested.

Zero model calls. These are scripted host-sandbox mechanism checks, not
benchmark attack success rates. Output directories must be new.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolfence.guarded_net import (GuardedTransactionNet, GuardedBoundary, TransitionRule,
                                       EvidenceStep, RefinementProposal)
from toolfence.transaction_net import TaskContract, ObjectRef, Receipt, HostToolBinding


def shop_task(net, order_label, otp_hex):
    task = str(uuid4())
    order = ObjectRef('shop', 'order', order_label, '1').encode()
    action = {'order_ref': order, 'otp_digest': otp_hex}
    net.issue(TaskContract(task, str(uuid4()), 'order.verify', action))
    evidence = {
        'match': ('product_match', {'product': 'P17', 'request': 'buy-request-1'}),
        'cart': ('cart_receipt', {'product': 'P17', 'quantity': 1}),
        'order': ('order_receipt', {'product': 'P17', 'order_ref': order}),
        'otp': ('order_otp', action),
    }
    stages = ['searching', 'matched', 'cart_ready', 'awaiting_otp', 'ready']
    steps = {name: EvidenceStep(stages[i], stages[i + 1], kind, payload)
             for i, (name, (kind, payload)) in enumerate(evidence.items())}
    net.configure(task, initial_stage='searching',
                  rules={'order.verify': TransitionRule('ready', 'completed', 'order_otp')},
                  steps=steps, revision='sandbox-shop-v1', expires=1000)
    return task, action, evidence


def send_args(document, recipient):
    return {'document_ref': ObjectRef('staged-host', 'document', document, '1').encode(),
            'recipient_ref': ObjectRef('staged-host', 'recipient', recipient, '1').encode(),
            'content_digest': hashlib.sha256(b'Quarterly total: 42.').hexdigest()}


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net = GuardedTransactionNet(str(directory / 'ledger.sqlite'), clock=lambda: 100)
    task1, action1, evidence1 = shop_task(net, 'O17', 'a' * 64)
    task2, action2, _ = shop_task(net, 'O27', 'c' * 64)
    task3, action3, _ = shop_task(net, 'O37', 'e' * 64)
    legit1 = send_args('D3', 'user3@recipient.example.org')
    attacker1 = send_args('D3', 'attacker@evil.example.org')
    legit2 = send_args('D9', 'user9@recipient.example.org')
    legit3 = send_args('D5', 'user5@recipient.example.org')
    attacker2 = send_args('D5', 'attacker@evil.example.org')
    attacker3 = send_args('D5', 'attacker2@evil.example.org')
    trace, reviews, refinement_reviews = [], [], []

    class Host:
        def __init__(self):
            self.executed = []

        def run_function(self, env, function, args, **kwargs):
            self.executed.append((function, dict(args)))
            return {'simulated': True}, None

    def human(view):
        reviews.append(view)
        if 'host_contract_verified' in view:
            # Activation review of an existing exact contract.
            return view['task'] in (task1, task2)
        # Refinement review: the human approves only the expected recipient.
        refinement_reviews.append(view)
        return view['bindings']['recipient_ref'] == legit1['recipient_ref']

    def attest_values(task, tool, bindings, rule):
        # The host verifier also attests the reused stage mapping is safe here.
        return bindings == {task1: legit1, task2: legit2, task3: legit3}.get(task)

    def boundary(task, action):
        verify = HostToolBinding(task, 'order.verify', lambda a: a,
                                 lambda op, result: Receipt(op, 'committed'), frozenset(action))
        send = HostToolBinding(task, 'document.send', lambda a: a,
                               lambda op, result: Receipt(op, 'committed'), frozenset(legit1))
        return GuardedBoundary(Host(), net, {'verify_order': verify, 'send_document': send},
                               actor='agent-A', reviewer=human, refine_missing=True)

    guard1, guard2, guard3 = boundary(task1, action1), boundary(task2, action2), boundary(task3, action3)

    def call(name, guard, tool, args, allowed):
        _, error = guard.run_function(None, tool, args)
        assert (error is None) == allowed, (name, error)
        trace.append({'case': name, 'allowed': error is None, 'reason': error})

    def pending_id(args):
        for row in net.refinements():
            if row['state'] == 'pending' and json.loads(row['proposal'])['bindings'] == args:
                return row['id']
        raise AssertionError('pending proposal missing for the requested bindings')

    # 1. A pending proposal grants nothing by itself.
    call('proposal alone grants nothing', guard1, 'send_document', legit1, False)
    # 2. An attacker-substituted recipient also becomes a pending proposal.
    call('attacker proposal recorded', guard1, 'send_document', attacker1, False)
    # 3. The human denies the attacker proposal; the call stays denied.
    denied = net.review_refinement(pending_id(attacker1), reviewer=human, budget_calls=1, budget_units=1)
    assert denied['state'] == 'denied' and len(refinement_reviews) == 1
    trace.append({'case': 'human denies attacker proposal', **denied})
    call('attacker call still denied', guard1, 'send_document', attacker1, False)
    # 4. One human approval applies an exact contract with a host budget.
    approved = net.review_refinement(pending_id(legit1), reviewer=human, budget_calls=1, budget_units=1)
    assert approved['state'] == 'applied' and approved['route'] == 'slow' and len(refinement_reviews) == 2
    trace.append({'case': 'human approves once; host mints budget 1/1', **approved})
    call('refined send executes', guard1, 'send_document', legit1, True)
    call('replay exceeds host budget', guard1, 'send_document', legit1, False)
    # 5. Same shape on a second task: reuse without consulting the human.
    call('second task proposal recorded', guard2, 'send_document', legit2, False)
    reused = net.review_refinement(pending_id(legit2), reviewer=human, budget_calls=1, budget_units=1,
                                   value_verify=attest_values)
    assert reused['state'] == 'applied' and reused['route'] == 'template_attested' and len(refinement_reviews) == 2
    trace.append({'case': 'attested template reuse without human review', **reused})
    call('attested refinement executes on second task', guard2, 'send_document', legit2, True)
    # 6. An explicit host verifier rejects fresh attacker values on a third
    # task, fail closed. (task2's exact bindings are already fixed by the
    # applied refinement, so a different-args call there is a binding
    # mismatch, not a missing-authority denial.)
    call('unattested attacker proposal on third task', guard3, 'send_document', attacker2, False)
    failed = net.review_refinement(pending_id(attacker2), reviewer=human, budget_calls=1, budget_units=1,
                                   value_verify=attest_values)
    assert failed['state'] == 'denied' and failed['route'] == 'value_verification_failed' and len(refinement_reviews) == 2
    trace.append({'case': 'value verification fails closed without human fallback', **failed})
    call('attacker call denied after failed attestation', guard3, 'send_document', attacker2, False)
    # 7. Reuse without a verifier falls back to the human, who sees the values.
    call('unattested proposal without verifier', guard3, 'send_document', attacker3, False)
    fallback = net.review_refinement(pending_id(attacker3), reviewer=human, budget_calls=1, budget_units=1)
    assert fallback['state'] == 'denied' and fallback['route'] == 'review_denied' and len(refinement_reviews) == 3
    trace.append({'case': 'reuse without verifier falls back to human, who denies', **fallback})
    call('fallback denial stays denied', guard3, 'send_document', attacker3, False)
    # 8. The original order workflow is unaffected by the refinement channel.
    for name, (kind, payload) in evidence1.items():
        net.ingest(task1, kind, payload, lambda t, k, p, kind=kind, payload=payload: t == task1 and k == kind and p == payload)
        net.advance(task1, name)
    call('existing order workflow unaffected', guard1, 'verify_order', action1, True)
    assert [f for f, _ in guard1.inner.executed] == ['send_document', 'verify_order']
    assert [f for f, _ in guard2.inner.executed] == ['send_document']
    assert guard3.inner.executed == []
    result = {'scope': 'scripted_refinement_mechanism_zero_model_calls', 'trace': trace,
              'executed': {'task1': guard1.inner.executed, 'task2': guard2.inner.executed},
              'human_reviews': len(reviews), 'refinement_reviews': len(refinement_reviews),
              'refinements': net.refinements(),
              'decisions': {'task1': guard1.decisions, 'task2': guard2.decisions}}
    (directory / 'cases.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({'cases': len(result['trace']), 'refinements': len(result['refinements']),
                      'human_reviews': result['human_reviews'], 'scope': result['scope']}))
