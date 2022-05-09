import inspect
import pytest
import sys

from padl.dumptools import inspector, ast_utils
from padl.dumptools.sourceget import get_source


toplevel_callinfo = inspector.CallInfo('here')
toplevel_frame = inspect.currentframe()


def nested_frame(a, b, c=3, **kwargs):
    f = inspect.stack()[0].frame
    return inspector._get_scope_from_frame(f, 0)


def chain_nested(x=1):
    return nested_frame(x, 1)


def nested_builder():
    return inspector.CallInfo('here')


def double_nested_builder():
    def inner_builder():
        return inspector.CallInfo('here')
    return inner_builder()


class TestCallInfo:
    def test_toplevel(self):
        assert toplevel_callinfo.module.__name__ == __name__
        assert toplevel_callinfo.scope.scopelist == []
        assert toplevel_callinfo.function == '<module>'

    def test_nested(self):
        nested_callinfo = nested_builder()
        assert nested_callinfo.module.__name__ == __name__
        assert len(nested_callinfo.scope) == 1
        assert nested_callinfo.scope.scopelist[0][0] == 'nested_builder'
        assert nested_callinfo.function == 'nested_builder'

    def test_double_nested(self):
        with pytest.raises(AssertionError):
            double_nested_builder()


class Test_GetScopeFromFrame:
    def test_toplevel(self):
        scope = inspector._get_scope_from_frame(toplevel_frame, 0)
        assert scope.is_global()
        assert scope == 'Scope[tests.unittests.dumptools.test_inspector]'

    def test_nested(self):
        e = 1
        scope = nested_frame(e, 2, u=123)
        assert scope.def_source == get_source(__file__)
        assigns = scope.scopelist[0][1].body[:4]
        unparsed = [ast_utils.unparse(x).strip() for x in assigns]
        assert 'a = e' in unparsed
        assert 'b = 2' in unparsed
        assert 'c = 3' in unparsed
        assert "kwargs = {'u': 123}" in unparsed
        assert all(x.value._scope == 'Scope[tests.unittests.dumptools.test_inspector.test_nested]'
                   for x in assigns)

    def test_chain_nested(self):
        e = 1
        scope = chain_nested(e)
        assert scope == \
            'Scope[tests.unittests.dumptools.test_inspector.nested_frame]'
        chain_scope = scope.scopelist[0][1].body[0].value._scope
        assert chain_scope == \
            'Scope[tests.unittests.dumptools.test_inspector.chain_nested]'
        outer_scope = chain_scope.scopelist[0][1].body[0].value._scope
        assert outer_scope == \
            'Scope[tests.unittests.dumptools.test_inspector.test_chain_nested]'
        assign = outer_scope.scopelist[0][1].body[0]
        assert ast_utils.unparse(assign).strip() == 'e = 1'
        assign = chain_scope.scopelist[0][1].body[0]
        assert ast_utils.unparse(assign).strip() == 'x = e'
        assign = scope.scopelist[0][1].body[0]
        assert ast_utils.unparse(assign).strip() == 'a = x'

    def test_scope_too_long_raises(self):
        def x():
            def y():
                def z():
                    f = inspect.stack()[0].frame
                    return inspector._get_scope_from_frame(f, 0)
                return z
            return y

        with pytest.raises(AssertionError):
            x()()()


class A:
    def __init__(self):
        self.frameinfo = inspector.non_init_caller_frameinfo()


class B(A):
    def __init__(self):
        super().__init__()


class TestNonInitCallerFrameinfo:
    def test_trivial(self):
        here = inspect.stack()[0]
        there = inspector.non_init_caller_frameinfo()
        assert here.filename == there.filename
        assert here.function == there.function

    def test_class_init(self):
        here = inspect.stack()[0]
        there = A().frameinfo
        assert here.filename == there.filename
        assert here.function == there.function

    def test_child_class_init(self):
        here = inspect.stack()[0]
        there = B().frameinfo
        assert here.filename == there.filename
        assert here.function == there.function


class TestTraceThis:
    def test_works(self):
        import sys
        if sys.gettrace() is not None and 'coverage' in str(sys.gettrace()):
            return  # this doesn't work with coverage -- disabling

        def tracefunc(frame, event, arg):
            if event == 'return':
                assert arg == 123
                assert 0, 'this works'

        def to_trace(x):
            inspector.trace_this(tracefunc)
            return x

        with pytest.raises(AssertionError) as exc_info:
            to_trace(123)

        exc_info.match('this works')


class Test_InstructionsUpToCall:
    def test_from_string(self):
        s = 'b = 1; a(); f = 123'
        ix = inspector._instructions_up_to_call(s)
        assert [i.opname for i in ix] == ['LOAD_CONST', 'STORE_NAME', 'LOAD_NAME', 'CALL_FUNCTION']

    def test_from_code(self):
        def f():
            return inspector._instructions_up_to_call(inspect.currentframe().f_back.f_code)

        a = 1
        b = 2
        ix = f()

        assert ix[-1].opname == 'CALL_FUNCTION'


class Test_InstructionsInName:
    def test_from_string(self):
        s = 'a.b.c.d'
        ix = inspector._instructions_in_name(s)
        assert [i.opname for i in ix] == ['LOAD_NAME', 'LOAD_ATTR', 'LOAD_ATTR', 'LOAD_ATTR']


class Test_InstructionsInGetitem:
    ...


class Test_InstructionsUpToOffset:
    def test_from_string(self):
        s = 'b = 1; a(); f = 123'
        ix = inspector._instructions_up_to_offset(s, 6)
        assert ix[-1].offset == 6
        ix = inspector._instructions_up_to_offset(s, 2)
        assert ix[-1].offset == 2


class Test_Module:
    def test_works(self):
        m = inspector._module(toplevel_frame)
        assert m == sys.modules[__name__]


class TestOuterCallerFrameinfo:
    def test_here(self):
        from tests.material import outer_caller
        m = outer_caller.here()
        assert m.function == 'test_here'

    def test_there(self):
        from tests.material import outer_caller_1
        m = outer_caller_1.there()
        assert m.function == 'there'


class TestCallerModule:
    def test_here(self):
        from tests.material import outer_caller
        assert outer_caller.caller_module().__name__ == __name__


class TestCallerFrame:
    def test_here(self):
        from tests.material import outer_caller
        assert outer_caller.caller_frame().f_code.co_filename == __file__
