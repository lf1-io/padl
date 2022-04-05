import ast
from textwrap import dedent

import pytest

from padl.dumptools import symfinder, ast_utils, sourceget


def test_annassign():
    source = dedent('''
        a: int = 1
    ''')
    scoped_name = symfinder.ScopedName('a', None)
    code, node, name = symfinder.find_scopedname_in_source(scoped_name, source)
    assert code == 'a: int = 1'
    assert isinstance(node, ast.AnnAssign)
    assert name.name == 'a'


class TestFindScopedNameInSource:
    def test_simple(self):
        source = dedent('''
            a = 1
            b = a
            a = b
            ''')
        scoped_name = symfinder.ScopedName('a', '__main__')
        snippet, node, name = symfinder.find_scopedname_in_source(scoped_name, source)
        assert snippet == 'a = b'
        assert isinstance(node, ast.Assign)
        assert name.scope == '__main__'
        pos = ast_utils.get_position(source, node)
        assert name.pos == (pos.lineno, pos.col_offset)

    def test_above_line(self):
        source = dedent('''
            a = 1
            b = a
            a = b
            ''')
        scoped_name = symfinder.ScopedName('a', '__main__', (4, 0))
        snippet, node, name = symfinder.find_scopedname_in_source(scoped_name, source)
        assert snippet == 'a = 1'
        assert isinstance(node, ast.Assign)
        assert name.scope == '__main__'
        pos = ast_utils.get_position(source, node)
        assert name.pos == (pos.lineno, pos.col_offset)


class TestFindScopedNameInIpython:
    def test_simple(self, monkeypatch):
        def _ipython_history():
            return [
                dedent('''
                    a = 1
                    b = a
                '''),
                dedent('''
                    a = b
                ''')
            ]
        monkeypatch.setattr(sourceget, '_ipython_history', _ipython_history)
        scoped_name = symfinder.ScopedName('a', symfinder.Scope.toplevel('__main__'))
        snippet, node, name = symfinder.find_scopedname_in_ipython(scoped_name)
        assert snippet == 'a = b'
        assert isinstance(node, ast.Assign)
        assert name.scope == symfinder.Scope.toplevel('__main__')
        pos = ast_utils.get_position(_ipython_history()[1], node)
        assert name.pos == (pos.lineno, pos.col_offset)
        assert name.cell_no == 1

    def test_second_cell(self, monkeypatch):
        def _ipython_history():
            return [
                dedent('''
                    a = 1
                    b = a
                '''),
                dedent('''
                    a = b
                ''')
            ]
        monkeypatch.setattr(sourceget, '_ipython_history', _ipython_history)
        scoped_name = symfinder.ScopedName('a', symfinder.Scope.toplevel('__main__'), pos=(2, 0), cell_no=1)
        snippet, node, name = symfinder.find_scopedname_in_ipython(scoped_name)
        assert snippet == 'a = 1'
        assert isinstance(node, ast.Assign)
        assert name.scope == symfinder.Scope.toplevel('__main__')
        pos = ast_utils.get_position(_ipython_history()[0], node)
        assert name.pos == (pos.lineno, pos.col_offset)
        assert name.cell_no == 0


class Test_GetCallSignature:
    def test_pos_args(self):
        pos, kw, star_args, star_kwargs = symfinder._get_call_signature('f(1, 2, "3", x)')
        assert pos == ['1', '2', '"3"', 'x']

    def test_kw_args(self):
        pos, kw, star_args, star_kwargs = symfinder._get_call_signature('f(x=1, y=2, z="3", a=x)')
        assert kw == {'x': '1', 'y': '2', 'z': '"3"', 'a': 'x'}

    def test_pos_and_kw_args(self):
        pos, kw, star_args, star_kwargs = symfinder._get_call_signature('f(1, 2, "3", x, x=1, y=2, z="3", a=x)')
        assert pos == ['1', '2', '"3"', 'x']
        assert kw == {'x': '1', 'y': '2', 'z': '"3"', 'a': 'x'}

    def test_star_args(self):
        pos, kw, star_args, star_kwargs = symfinder._get_call_signature('f(*[1, 2, 3])')
        assert star_args == '[1, 2, 3]'

    def test_star_kwargs(self):
        pos, kw, star_args, star_kwargs = symfinder._get_call_signature('f(**{"a": 1})')
        assert star_kwargs == '{"a": 1}'
