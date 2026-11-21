"""Reproducible host-sandbox cases for the unified guarded net; zero model calls."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolfence.guarded_net import GuardedTransactionNet, GuardedBoundary, TransitionRule, EvidenceStep
from toolfence.transaction_net import TaskContract, ObjectRef, Receipt, HostToolBinding


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net = GuardedTransactionNet(str(directory / 'ledger.sqlite'), clock=lambda: 100)
    task = str(uuid4())
    order = ObjectRef('shop', 'order', 'O17', '1').encode()
    action = {'order_ref': order, 'otp_digest': 'a' * 64}
    net.issue(TaskContract(task, str(uuid4()), 'order.verify', action))
    evidence = {
        'match': ('product_match', {'product': 'P17', 'request': 'buy-request-1'}),
        'cart': ('cart_receipt', {'product': 'P17', 'quantity': 1}),
        'order': ('order_receipt', {'product': 'P17', 'order_ref': order}),
        'otp': ('order_otp', action),
    }
    stages = ['searching', 'matched', 'cart_ready', 'awaiting_otp', 'ready']
    steps = {name: EvidenceStep(stages[i], stages[i+1], kind, payload)
             for i, (name, (kind, payload)) in enumerate(evidence.items())}
    net.configure(task, initial_stage='searching', rules={'order.verify': TransitionRule('ready', 'completed', 'order_otp')},
                  steps=steps, revision='sandbox-shop-v1', expires=1000)
    executed, reviews, trace = [], [], []
    class Shop:
        def run_function(self, env, function, args, **kwargs):
            executed.append(dict(args))
            return {'order': 'O17', 'verified': True}, None
    def review(proposal):
        reviews.append(proposal)
        return True
    binding = HostToolBinding(task, 'order.verify', lambda a: a,
                             lambda op, result: Receipt(op, 'committed'), frozenset(action))
    guard = GuardedBoundary(Shop(), net, {'verify_order': binding}, actor='agent-A', reviewer=review)
    def call(name, args, allowed):
        _, error = guard.run_function(None, 'verify_order', args)
        assert (error is None) == allowed, (name, error)
        trace.append({'case': name, 'allowed': error is None, 'reason': error, 'marking': net.marking(task)})
    call('skip shopping and OTP prerequisites', action, False)
    for name, (kind, payload) in evidence.items():
        # Attestor compares against pinned sandbox records, not model text.
        net.ingest(task, kind, payload, lambda t, k, p, kind=kind, payload=payload: t == task and k == kind and p == payload)
        net.advance(task, name)
    call('injected order substitution O99', {**action, 'order_ref': ObjectRef('shop', 'order', 'O99', '1').encode()}, False)
    call('wrong OTP digest', {**action, 'otp_digest': 'b' * 64}, False)
    call('authorized O17 verification', action, True)
    call('replay after success', action, False)
    assert executed == [action]
    result = {'scope': 'scripted_host_sandbox_not_model_benchmark', 'trace': trace,
              'executed': executed, 'review_calls': len(reviews), 'decisions': guard.decisions}
    (directory / 'cases.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({'cases': len(result['trace']), 'executed': len(result['executed']), 'scope': result['scope']}))
