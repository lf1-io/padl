import ast
import pytest
from textwrap import dedent

from padl.dumptools import var2mod
from padl.dumptools.var2mod import ScopedName


class TestFindGlobals:
    def test_find_same_name(self):
        statement = 'a = run(a)'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {ScopedName('a', None), ScopedName('run', None)}

    def test_find_in_assignment(self):
        statement = 'a = run'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {ScopedName('run', None)}

    def test_dots(self):
        statement = (
            'def f(x):\n'
            '    o = x.y\n'
            '    u = y.x\n'
            '    return a.b.c(x)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('a.b.c', None), ScopedName('y.x', None)}

    def test_attribute(self):
        statement = (
            'def f(x):\n'
            '    o = x.y\n'
            '    u = y.x\n'
            '    ff = (aa + bb).c\n'
            '    return a.b.c(x)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('a.b.c', None), ScopedName('y.x', None),
                       ScopedName('aa', None), ScopedName('bb', None)}

    def test_complex_statement_1(self):
        statement = (
            '@transform\n'
            'def f(x):\n'
            '    (255 * (x * 0.5 + 0.5)).numpy().astype(numpy.uint8)\n'
        )
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('numpy.uint8', None), ScopedName('transform', None)}


class TestRename:
    source = dedent('''\
        aaa = 1

        def ooo():
            o = aaa  # global - should be renamed

        class CCC:
            o = aaa

            def __init__(self, aaa):  # local - shouldn't be renamed
                self.aaa = aaa  # local

            def aaa(self):  # local
                self.aaa = 123  # attribute - shoudn't be renamed

            def bbb(self):
                return aaa + self.aaa  # global - should be renamed

        def bbb(aaa):  # local
            aaa = 1
            def ccc(bbb):
                return 1 + aaa

        def ccc():
            def ddd():
                return aaa  # global
            return ddd

        def fff(aaa, bbb=aaa):  # default is global
            return x
        ''')

    target = dedent('''\
        xxx = 1

        def ooo():
            o = xxx  # global - should be renamed

        class CCC:
            o = xxx

            def __init__(self, aaa):  # local - shouldn't be renamed
                self.aaa = aaa  # local

            def aaa(self):  # local
                self.aaa = 123  # attribute - shoudn't be renamed

            def bbb(self):
                return xxx + self.aaa  # global - should be renamed

        def bbb(aaa):  # local
            aaa = 1
            def ccc(bbb):
                return 1 + aaa

        def ccc():
            def ddd():
                return xxx  # global
            return ddd

        def fff(aaa, bbb=xxx):  # default is global
            return x
        ''')

    def test_rename(self):
        renamed = var2mod.rename(self.source, from_='aaa', to='xxx')
        assert renamed == self.target
