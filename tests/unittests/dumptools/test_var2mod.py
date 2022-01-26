import ast

from padl.dumptools import var2mod
from padl.dumptools.var2mod import ScopedName


class TestFindGlobals:
    def test_find_same_name(self):
        statement = 'a = run(a)'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {ScopedName('a', None, 1), ScopedName('run', None, 0)}

    def test_find_in_assignment(self):
        statement = 'a = run'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {ScopedName('run', None, 0)}

    def test_dots(self):
        statement = (
            'def f(x):\n'
            '    o = x.y\n'
            '    u = y.x\n'
            '    return a.b.c(x)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('a.b.c', None, 0), ScopedName('y.x', None, 0)}

    def test_attribute(self):
        statement = (
            'def f(x):\n'
            '    o = x.y\n'
            '    u = y.x\n'
            '    ff = (aa + bb).c\n'
            '    return a.b.c(x)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('a.b.c', None, 0), ScopedName('y.x', None, 0),
                       ScopedName('aa', None, 0), ScopedName('bb', None, 0)}

    def test_complex_statement_1(self):
        statement = (
            '@transform\n'
            'def f(x):\n'
            '    (255 * (x * 0.5 + 0.5)).numpy().astype(numpy.uint8)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('numpy.uint8', None, 0), ScopedName('transform', None, 0)}
