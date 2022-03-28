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


class TestUpdateScopedName:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.scoped_name_1 = symfinder.ScopedName('a', None, 1)
        request.cls.scoped_name_1.add_variant('a.B.C', 1)

        request.cls.scoped_name_2 = symfinder.ScopedName('b', None, 2)
        request.cls.scoped_name_2.add_variant('b.T', 2)

    def test_update_scopedname_add_n(self):
        return_scoped_name = symfinder.update_scopedname(self.scoped_name_1, self.scoped_name_1.scope,
                                                         2, remove_dot=False)

        returned_names, returned_ns = list(zip(*return_scoped_name.variants))

        assert all([name1 == name2 for name1, name2 in zip(['a', 'a.B.C'], returned_names)])
        assert all([n == 3 for n in returned_ns])

    def test_update_scopedname_remove_dot(self):
        return_scoped_name = symfinder.update_scopedname(self.scoped_name_1, self.scoped_name_1.scope,
                                                         remove_dot=True)

        returned_names, returned_ns = list(zip(*return_scoped_name.variants))

        assert all([name1 == name2 for name1, name2 in zip(['a', 'a.B'], returned_names)])
        assert all([n == 1 for n in returned_ns])

    def test_update_scopedname_add_n_remove_dot(self):
        return_scoped_name = symfinder.update_scopedname(self.scoped_name_2, self.scoped_name_2.scope,
                                                         add_n=2, remove_dot=True)

        returned_names, returned_ns = list(zip(*return_scoped_name.variants))

        assert all([name1 == name2 for name1, name2 in zip(['b', 'b'], returned_names)])
        assert all([n == 4 for n in returned_ns])
