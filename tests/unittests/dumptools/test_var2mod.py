import ast

from padl.dumptools import var2mod


class TestFindGlobals:
    def test_find_same_name(self):
        statement = 'a = run(a)'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {('a', 1), ('run', 0)}

    def test_find_in_assignment(self):
        statement = 'a = run'
        tree = ast.parse(statement)
        res = var2mod.find_globals(tree)
        assert res == {('run', 0)}
