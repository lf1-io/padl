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

    scoped_name = symfinder.ScopedName('a', None, 0)
    code, node, name = symfinder.find_scopedname_in_source(scoped_name, source)
    assert code == 'a: int = 1'
    assert isinstance(node, ast.AnnAssign)
    assert name == 'a'
