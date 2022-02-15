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


testcode_0 = '''
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


testcode_1 = '''
def c(one=1
      two=2):
    """
    this is a comment
    """
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


testcode_2 = '''
def x(one=1,
      two=2):
    return one + two
'''


testcode_3 = '''
def x(one=1,
    two=2):
    return one + two
'''


testcode_4 = '''
def x(one=1,
    two=2) -> int:
    return one + two
'''


class TestGetStatement:
    def test_simple_1(self):
        statement, _ = inspector.get_statement(testcode_0, 2)
        assert statement == 'a = 1'

    def test_simple_2(self):
        statement, _ = inspector.get_statement(testcode_0, 3)
        assert statement == 'b = 2'

    def test_function(self):
        statement, _ = inspector.get_statement(testcode_0, 5)
        assert statement == testcode_0.split(' = 2')[1].strip()

    def test_multiline_1(self):
        statement, _ = inspector.get_statement(testcode_0, 7)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_2(self):
        statement, _ = inspector.get_statement(testcode_0, 8)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_3(self):
        statement, _ = inspector.get_statement(testcode_0, 9)
        assert statement == 'p = (a,\n    b,\n    c)'

    def test_multiline_4(self):
        statement, _ = inspector.get_statement(testcode_0, 11)
        assert statement == 'f = a +         b'

    def test_multiline_5(self):
        statement, _ = inspector.get_statement(testcode_0, 14)
        assert statement == 'k = f(1,\n      2, 3,\n      4)'


class TestGetSurroundingBlock:
    def test_hanging_arguments_1(self):
        block, l, r = inspector.get_surrounding_block(testcode_1, 7)
        assert block.strip().startswith('"""')

    def test_hanging_arguments_2(self):
        block, l, r = inspector.get_surrounding_block(testcode_1, 12)
        assert block.strip().startswith('"""')

    def test_hanging_arguments_3(self):
        block, l, r = inspector.get_surrounding_block(testcode_2, 4)
        assert block.strip().startswith('return')

    def test_hanging_arguments_4(self):
        block, l, r = inspector.get_surrounding_block(testcode_3, 4)
        assert block.strip().startswith('return')

    def test_hanging_arguments_5(self):
        block, l, r = inspector.get_surrounding_block(testcode_4, 4)
        assert block.strip().startswith('return')
