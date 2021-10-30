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


testcode = '''
a = 1
b = 2

def c():
    o = 7
    p = (a,
        b,
        c)

    f = a + \
        b

    k = f(1,
          2, 3,
          4)
    return f
'''


class TestGetStatement:
    def test_simple(self):
        statement, _ = inspector.get_statement(testcode, 2)
        assert statement == 'a = 1'
        statement, _ = inspector.get_statement(testcode, 3)
        assert statement == 'b = 2'

    def test_function(self):
        statement, _ = inspector.get_statement(testcode, 5)
        assert statement == testcode.split(' = 2')[1].strip()

    def test_multiline_1(self):
        statement, _ = inspector.get_statement(testcode, 7)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_2(self):
        statement, _ = inspector.get_statement(testcode, 8)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_3(self):
        statement, _ = inspector.get_statement(testcode, 9)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_4(self):
        statement, _ = inspector.get_statement(testcode, 11)
        assert statement == 'f = a +         b'

    def test_multiline_5(self):
        statement, _ = inspector.get_statement(testcode, 14)
        assert statement == 'k = f(1,\n      2, 3,\n      4)'
