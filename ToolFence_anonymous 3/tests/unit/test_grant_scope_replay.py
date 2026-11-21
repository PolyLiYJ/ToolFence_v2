import pytest
from toolfence.grant_replay import Scope, decide, Incomplete


def snapshot():
    return dict(origin="approved_runtime_grant", grant=dict(tool="read", bindings={"file":"user"}, stage="ready"),
                call=dict(tool="read", arguments={"file":"one"}, sources={"file":"user"}),
                state=dict(stage="ready", evidence={"user":["one"]}, remaining=1))


@pytest.mark.parametrize("field,expected", [
    ("valid",[1,1,1,1]), ("tool",[0,0,0,0]), ("file",[1,0,0,0]),
    ("source",[1,0,0,0]), ("stage",[1,1,0,0]), ("budget",[1,1,1,0])])
def test_final_decisions(field,expected):
    r = snapshot()
    if field == "tool": r["call"]["tool"] = "other"
    if field == "file": r["call"]["arguments"]["file"] = "two"
    if field == "source": r["call"]["sources"]["file"] = "other"
    if field == "stage": r["state"]["stage"] = "early"
    if field == "budget": r["state"]["remaining"] = 0
    assert [decide(r,s) for s in Scope] == expected


def test_blueprint_and_incomplete():
    assert all(decide(dict(origin="blueprint",blueprint_allowed=True),s) for s in Scope)
    r = snapshot()
    del r["state"]["remaining"]
    with pytest.raises(Incomplete):
        decide(r,Scope.TOOL)
