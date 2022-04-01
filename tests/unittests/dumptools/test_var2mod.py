import ast

import pytest

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


class Test_CheckAndMakeVariants:
    @staticmethod
    def parse(eqn):
        t, v = eqn.split('=')
        target = var2mod.ScopedName(full_name=t.strip(), n=0, scope=None)
        value = var2mod.ScopedName(full_name=v.strip(), n=0, scope=None)
        return [target], value

    def test_a_a(self):
        res = var2mod._increment_variants_from_targets(*self.parse('a = a'))
        assert set(res.variants().items()) == {('a', 1)}

    def test_a_b(self):
        res = var2mod._increment_variants_from_targets(*self.parse('a = b'))
        assert set(res.variants().items()) == {('b', 0)}

    def test_a_ab(self):
        res = var2mod._increment_variants_from_targets(*self.parse('a = a.b'))
        assert set(res.variants().items()) == {('a', 1), ('a.b', 0)}

    def test_a_abc(self):
        res = var2mod._increment_variants_from_targets(*self.parse('a = a.b.c'))
        assert set(res.variants().items()) == {('a', 1), ('a.b', 0), ('a.b.c', 0)}
