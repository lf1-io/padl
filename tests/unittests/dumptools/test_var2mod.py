import ast
import builtins
from collections import defaultdict
from textwrap import dedent

import pytest

from padl.dumptools import var2mod
from padl.dumptools.var2mod import ScopedName


SOURCE = dedent('''\
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

    o = [aaa for aaa in bla]  # local
    ''')

TARGET = dedent('''\
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

    o = [aaa for aaa in bla]  # local
    ''')


class TestFinder:
    def test_find_function_def_explicit(self):
        res = var2mod.Finder(ast.FunctionDef).find(ast.parse(SOURCE))
        assert all(isinstance(x, ast.FunctionDef) for x in res)
        assert {x.name for x in res} == {'ooo', '__init__', 'aaa', 'bbb', 'ccc', 'fff', 'ddd'}

    def test_find_all(self):
        parsed = ast.parse(SOURCE)
        all_nodes = list(ast.walk(parsed))
        filtered = defaultdict(set)
        for x in all_nodes:
            filtered[type(x)].add(x)

        for type_, instances in filtered.items():
            res = var2mod.Finder(type_).find(parsed)
            assert set(res) == instances


class Test_JoinAttr:
    def test_attribute_chain(self):
        assert var2mod._join_attr(ast.parse('a.b.c.d.e.f.g').body[0].value) == \
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def test_single_name(self):
        assert var2mod._join_attr(ast.parse('a').body[0].value) == ['a']

    def test_raises(self):
        with pytest.raises(TypeError):
            var2mod._join_attr(ast.parse('f()').body[0].value)


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

    def test_nested_function(self):
        statement = dedent('''\
            def f():
                def g():
                    x = aaa
        ''')
        res = var2mod.find_globals(ast.parse(statement))
        assert res == {ScopedName('aaa', None)}

    def test_nested_function_nonlocal(self):
        statement = dedent('''\
            def f():
                def g():
                    x = aaa
                aaa = 1
        ''')
        res = var2mod.find_globals(ast.parse(statement))
        assert res == set()


class TestRename:

    def test_complex_example(self):
        renamed = var2mod.rename(SOURCE, from_='aaa', to='xxx')
        assert renamed == TARGET

    def test_assignment_1(self):
        source = 'aaa = aaa'
        target = 'xxx = xxx'
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_assignment_2(self):
        source = 'aaa = bbb'
        target = 'xxx = bbb'
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_assignment_3(self):
        source = 'bbb = aaa'
        target = 'bbb = xxx'
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_function_body(self):
        source = dedent('''\
            def f():
                bbb = aaa
        ''')
        target = dedent('''\
            def f():
                bbb = xxx
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_function_name(self):
        source = dedent('''\
            def aaa():
                ...
        ''')
        target = dedent('''\
            def xxx():
                ...
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_function_name_no_locals(self):
        source = dedent('''\
            def aaa():
                ...
        ''')
        target = dedent('''\
            def aaa():
                ...
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx', rename_locals=False)
        assert renamed == target

    def test_nested_function_body(self):
        source = dedent('''\
            def f():
                def g():
                    bbb = aaa
        ''')
        target = dedent('''\
            def f():
                def g():
                    bbb = xxx
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_comprehension(self):
        source = 'bbb = [aaa for aaa in bla]'
        target = 'bbb = [aaa for aaa in bla]'
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_class_name(self):
        source = dedent('''\
            class aaa():
                ...
        ''')
        target = dedent('''\
            class xxx():
                ...
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target

    def test_class_name_no_locals(self):
        source = dedent('''\
            class aaa():
                ...
        ''')
        target = dedent('''\
            class aaa():
                ...
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx', rename_locals=False)
        assert renamed == target

    def test_class_body(self):
        source = dedent('''\
            class bbb:
                ooo = aaa
        ''')
        target = dedent('''\
            class bbb:
                ooo = xxx
        ''')
        renamed = var2mod.rename(source, from_='aaa', to='xxx')
        assert renamed == target


class TestAddScopeAndPos:
    def test_only_change_empty_scope(self):
        scope1 = var2mod.Scope.toplevel(var2mod)
        scope2 = var2mod.Scope.toplevel(ast)
        vars = [var2mod.ScopedName('x', scope=scope1)]
        name = var2mod.ScopedName('y', scope=scope2)
        var2mod.add_scope_and_pos(vars, name, ast.Name())
        assert [var.scope == scope1 for var in vars]

    def test_add_scope(self):
        scope2 = var2mod.Scope.toplevel(ast)
        vars = [var2mod.ScopedName('x')]
        name = var2mod.ScopedName('y', scope=scope2)
        var2mod.add_scope_and_pos(vars, name, ast.Name())
        assert [var.scope == scope2 for var in vars]

    def test_add_pos(self):
        vars = [var2mod.ScopedName('x')]
        name = var2mod.ScopedName('y', pos=(1, 1))
        var2mod.add_scope_and_pos(vars, name, ast.Name())
        assert [var.pos == (1, 1) for var in vars]


class Test_MethodFinder:
    def test_simple(self):
        source = dedent('''\
            class X:
                def a(self):
                    ...

                def b(self):
                    ...

                def c(self):
                    ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}

    def test_classmethod(self):
        source = dedent('''\
            class X:
                @classmethod
                def a(self):
                    ...

                @classmethod
                def b(cls):
                    ...

                @classmethod
                def c(cls):
                    ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}

    def test_staticmethod(self):
        source = dedent('''\
            class X:
                @staticmethod
                def a(self):
                    ...

                @staticmethod
                def b(cls):
                    ...

                @staticmethod
                def c(cls):
                    ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}

    def test_mixed(self):
        source = dedent('''\
            class X:
                def a(self):
                    ...

                @staticmethod
                def b(cls):
                    ...

                @classmethod
                def c(cls):
                    ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}

    def test_nested_function(self):
        source = dedent('''\
            class X:
                def a(self):
                    def aa():
                        ...

                @staticmethod
                def b(cls):
                    def bb():
                        ...

                @classmethod
                def c(cls):
                    def cc():
                        ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}

    def test_nested_class(self):
        source = dedent('''\
            class X:
                def a(self):
                    ...

                @staticmethod
                def b(cls):
                    ...

                @classmethod
                def c(cls):
                    ...

                class X:
                    def oh(self):
                        ...
        ''')
        res = var2mod._MethodFinder().find(ast.parse(source).body[0])
        assert set(res.keys()) == {'a', 'b', 'c'}


class Test_FilterBuiltins:
    def test_works(self):
        names = [var2mod.ScopedName(x) for x in builtins.__dict__.keys()]
        names.insert(0, var2mod.ScopedName('bla'))
        names.insert(3, var2mod.ScopedName('ble'))
        names.insert(12, var2mod.ScopedName('bli'))
        names.insert(22, var2mod.ScopedName('blo'))
        names.append(var2mod.ScopedName('blu'))
        assert var2mod._filter_builtins(names) == {var2mod.ScopedName(x)
                                                   for x in {'bla', 'ble', 'bli', 'blo', 'blu'}}


class Test_FindGlobalsInClassdef:
    def test_finds_in_function_def(self):
        source = dedent('''\
            class A:
                def a(self):
                    o = bla

                def b(self):
                    o = ble

                def c(self, bli):
                    o = bli
        ''')
        res = var2mod._find_globals_in_classdef(ast.parse(source).body[0])
        assert {x.name for x in res} == {'bla', 'ble'}

    def test_finds_in_body(self):
        source = dedent('''\
            class A:
                x = bla
                y = [f for f in ble]
                blu = bla
        ''')
        res = var2mod._find_globals_in_classdef(ast.parse(source).body[0])
        assert {x.name for x in res} == {'bla', 'ble'}

    def test_keeps_builtins(self):
        source = dedent('''\
            class A:
                def a(self):
                    o = int

                def b(self):
                    o = ble

                def c(self, bli):
                    o = bli
        ''')
        res = var2mod._find_globals_in_classdef(ast.parse(source).body[0], filter_builtins=False)
        assert {x.name for x in res} == {'int', 'ble'}

    def test_filters_builtins(self):
        source = dedent('''\
            class A:
                def a(self):
                    o = int

                def b(self):
                    o = ble

                def c(self, bli):
                    o = bli
        ''')
        res = var2mod._find_globals_in_classdef(ast.parse(source).body[0])
        assert {x.name for x in res} == {'ble'}

    def test_nested_class(self):
        source = dedent('''\
            class A:
                class B:
                    def a(self):
                        o = bla

                    def b(self):
                        o = ble

                    def c(self, bli):
                        o = bli
        ''')
        res = var2mod._find_globals_in_classdef(ast.parse(source).body[0], filter_builtins=False)
        assert {x.name for x in res} == {'bla', 'ble'}


class TestFindCodenode:
    def test_find_toplevel(self):
        scope = var2mod.Scope.toplevel(var2mod)
        node = var2mod.ScopedName('find_codenode', scope=scope)
        res = var2mod.find_codenode(node)
        assert res.source.startswith('def find_codenode')
        assert isinstance(res.ast_node, ast.FunctionDef)
        assert res.ast_node.name == 'find_codenode'
        assert res.name.name == 'find_codenode'
        assert res.name.scope == scope
        assert res.name.pos is not None

    def test_find_from_import(self):
        from tests.material import find_codenode_from_import
        scope = var2mod.Scope.toplevel(find_codenode_from_import)
        node = var2mod.ScopedName('find_codenode', scope=scope)
        res = var2mod.find_codenode(node)
        assert res.source == 'from padl.dumptools.var2mod import find_codenode'
        assert isinstance(res.ast_node, ast.ImportFrom)
        assert res.name.name == 'find_codenode'
        assert res.name.scope == var2mod.Scope.empty()
        assert res.name.pos is not None

    def test_find_from_import_fulldump(self):
        from tests.material import find_codenode_from_import
        scope = var2mod.Scope.toplevel(find_codenode_from_import)
        node = var2mod.ScopedName('find_codenode', scope=scope)
        res = var2mod.find_codenode(node, full_dump_module_names='padl.dumptools.var2mod')
        assert res.source.startswith('def find_codenode')
        assert isinstance(res.ast_node, ast.FunctionDef)
        assert res.ast_node.name == 'find_codenode'
        assert res.name.name == 'find_codenode'
        assert res.name.scope == var2mod.Scope.toplevel(var2mod)
        assert res.name.pos is not None

    def test_find_module_import(self):
        from tests.material import find_codenode_module_import
        scope = var2mod.Scope.toplevel(find_codenode_module_import)
        node = var2mod.ScopedName('var2mod.find_codenode', scope=scope)
        res = var2mod.find_codenode(node)
        assert res.source == 'from padl.dumptools import var2mod'
        assert isinstance(res.ast_node, ast.ImportFrom)
        assert res.name.name == 'var2mod'
        assert res.name.scope == var2mod.Scope.empty()
        assert res.name.pos is not None
