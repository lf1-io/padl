import pytest

from padl.dumptools import inspector


toplevel_callinfo = inspector.CallInfo('here')


def nested_builder():
    return inspector.CallInfo('here')


def double_nested_builder():
    def inner_builder():
        return inspector.CallInfo('here')
    return inner_builder()


class TestCallerInfo:
    def test_toplevel(self):
        assert toplevel_callinfo.module.__name__ == __name__
        assert toplevel_callinfo.scope.scopelist == []

    def test_nested(self):
        nested_callinfo = nested_builder()
        assert nested_callinfo.module.__name__ == __name__
        assert len(nested_callinfo.scope) == 1
        assert nested_callinfo.scope.scopelist[0][0] == 'nested_builder'

    def test_double_nested(self):
        with pytest.raises(AssertionError):
            double_nested_builder()
