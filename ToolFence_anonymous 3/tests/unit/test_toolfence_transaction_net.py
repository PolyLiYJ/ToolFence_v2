from concurrent.futures import ThreadPoolExecutor
import json
import re
from uuid import uuid4

import pytest

from toolfence.transaction_net import TransactionNet, TaskContract, ObjectRef, Receipt
from toolfence.semantic.prompts import TOOL_SEMANTICS_SYSTEM_PROMPT, tool_semantics_prompt
from toolfence.semantic.schema import ToolManifest
from toolfence.semantic.architect import SemanticPolicyCompiler


def uid():
    return str(uuid4())


def payment(tmp_path):
    net = TransactionNet(str(tmp_path / 'ledger.sqlite'))
    task = uid()
    args = {'account_ref': ObjectRef('bank', 'account', 'alice', '1').encode(), 'amount_minor': 100, 'currency': 'USD'}
    net.issue(TaskContract(task, uid(), 'payment.transfer', args, max_units=100))
    return net, task, args


def test_prompt_examples_validate_against_actual_schema():
    examples = re.findall(r'Correct complete response:\n(.*?)(?=\n\n|\n"""|$)', TOOL_SEMANTICS_SYSTEM_PROMPT, re.S)
    assert len(examples) == 3
    for text in examples:
        ToolManifest.model_validate_json(text.strip())
    prompt = json.loads(tool_semantics_prompt([{'name': 'one', 'parameters': {}}]))
    assert prompt['required_tool_names'] == ['one']
    assert 'tools' in prompt['response_json_schema']['required']


def test_missing_name_must_be_repaired_by_model_not_guessed():
    responses = iter([{'tools': [{'effect': 'read'}]}, {'tools': [{'name': 'one', 'effect': 'read', 'operation': 'observe'}]}])
    prompts = []
    def backend(system, prompt):
        prompts.append(prompt)
        return json.dumps(next(responses))
    result = SemanticPolicyCompiler(backend).compile_manifest([{'name': 'one', 'parameters': {}}])
    assert result.tools[0].name == 'one'
    assert len(prompts) == 2
    assert 'name' in json.loads(prompts[1])['validation_error']


def test_reject_duplicate_catalog_before_model():
    compiler = SemanticPolicyCompiler(lambda *args: pytest.fail('must not call model'))
    with pytest.raises(ValueError, match='duplicate'):
        compiler.compile_manifest([{'name': 'one'}, {'name': 'one'}])


def test_concurrent_reservations_share_one_budget(tmp_path):
    net, task, args = payment(tmp_path)
    def reserve(_):
        try:
            return net.reserve(uid(), task, 'payment.transfer', args)
        except PermissionError:
            return 'denied'
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(16)))
    assert outcomes.count('reserved') == 1


def test_unknown_holds_budget_until_authenticated_reconciliation(tmp_path):
    net, task, args = payment(tmp_path)
    operation = uid()
    net.reserve(operation, task, 'payment.transfer', args)
    def dropped_response(*_):
        raise TimeoutError('response lost after side effect')
    with pytest.raises(TimeoutError):
        net.dispatch(operation, dropped_response)
    with pytest.raises(PermissionError):
        net.abort(operation)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'payment.transfer', args)
    assert net.reconcile(operation, lambda op: Receipt(op, 'not_applied')) == 'aborted'
    assert net.reserve(uid(), task, 'payment.transfer', args) == 'reserved'
    with pytest.raises(PermissionError):
        net.reconcile(operation, lambda op: Receipt(op, 'not_applied'))


def test_idempotency_replay_and_cross_task_denied(tmp_path):
    net, task, args = payment(tmp_path)
    operation = uid()
    net.reserve(operation, task, 'payment.transfer', args)
    assert net.dispatch(operation, lambda op, *_: Receipt(op, 'committed')) == 'committed'
    assert net.reserve(operation, task, 'payment.transfer', args) == 'committed'
    with pytest.raises(PermissionError):
        net.dispatch(operation, lambda *_: pytest.fail('repeated dispatch'))
    with pytest.raises(PermissionError):
        net.reserve(operation, uid(), 'payment.transfer', args)


def test_otp_bound_to_exact_order_and_task(tmp_path):
    net = TransactionNet(str(tmp_path / 'ledger.sqlite'))
    task = uid()
    args = {'order_ref': ObjectRef('shop', 'order', '42', '1').encode(), 'otp_digest': 'a' * 64}
    net.issue(TaskContract(task, uid(), 'order.verify', args))
    with pytest.raises(PermissionError):
        net.ingest(task, 'order_otp', args, lambda *_: False)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'order.verify', args)
    net.ingest(task, 'order_otp', args, lambda *_: True)
    with pytest.raises(PermissionError):
        net.reserve(uid(), task, 'order.verify', {**args, 'otp_digest': 'b' * 64})
    assert net.reserve(uid(), task, 'order.verify', args) == 'reserved'


def test_document_recipient_and_content_are_pinned(tmp_path):
    net = TransactionNet(str(tmp_path / 'ledger.sqlite'))
    task = uid()
    args = {'document_ref': ObjectRef('drive', 'document', 'report', '1').encode(), 'recipient_ref': ObjectRef('mail', 'recipient', 'alice', '1').encode(), 'content_digest': 'a' * 64}
    net.issue(TaskContract(task, uid(), 'document.send', args))
    for changed in ({**args, 'content_digest': 'b' * 64}, {**args, 'bcc': 'mallory'}):
        with pytest.raises(PermissionError):
            net.reserve(uid(), task, 'document.send', changed)
    op = uid()
    net.reserve(op, task, 'document.send', args)
    net.abort(op)
    assert net.reserve(uid(), task, 'document.send', args) == 'reserved'


def test_final_dispatch_boundary_consumes_budget_and_blocks_unmapped_tools(tmp_path):
    from toolfence.transaction_net import HostToolBinding, TransactionBoundary
    net, task, args = payment(tmp_path)
    executed = []
    class Inner:
        def run_function(self, env, function, kwargs, **_):
            executed.append((function, kwargs))
            return 'done', None
    binding = HostToolBinding(task, 'payment.transfer', lambda a: a, lambda op, result: Receipt(op, 'committed'), frozenset(args))
    runtime = TransactionBoundary(Inner(), net, {'send_money': binding})
    assert runtime.run_function(None, 'send_money', args) == ('done', None)
    assert runtime.run_function(None, 'send_money', args)[1]
    assert runtime.run_function(None, 'attacker_tool', {})[1].startswith('not_evaluated:')
    assert len(executed) == 1


def test_transactional_runtime_requires_host_contracts():
    from toolfence.colored_runtime import TransactionalColoredToolFenceRuntime
    with pytest.raises(ValueError, match='trusted transaction contracts'):
        TransactionalColoredToolFenceRuntime()
