import ast
from textwrap import dedent

import pytest

from padl.dumptools import symfinder


def test_annassign():
    source = dedent('''
        a: int = 1
    ''')
    code, node, name = symfinder.find_in_source('a', source)
    assert code == 'a: int = 1'
    assert isinstance(node, ast.AnnAssign)
    assert name == 'a'


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
