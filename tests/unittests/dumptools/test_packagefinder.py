import ast
import sys
from textwrap import dedent

from pytest import raises, warns
import torch

from padl.dumptools import packagefinder


class TestStandardLibNameGen:
    def test_works(self):
        res = list(packagefinder.standard_lib_names_gen())
        for x in res:
            assert isinstance(x, str)
            assert not x.startswith('_')
        assert 'dis' in res
        assert 'random' in res
        assert 'datetime' in res

    def test_include_underscored(self):
        res = list(packagefinder.standard_lib_names_gen(True))

        for x in res:
            assert isinstance(x, str)

        assert any(x.startswith('_') for x in res)


class TestGetDistributionName:
    def test_same_name(self):
        assert packagefinder.get_distribution_name('torch') == 'torch'

    def test_different_name(self):
        assert packagefinder.get_distribution_name('pytest_cov') == 'pytest-cov'

    def test_raises(self):
        with raises(packagefinder.RequirementNotFound):
            assert packagefinder.get_distribution_name('this_package_certainly_dosnt_exist')


class TestGetPackages:
    def test_different_imports(self):
        source = dedent('''
        import numpy as np
        import torch
        from ast import dump as dp
        from padl import dumptools
        from requests import *
        import ble.lo

        def x():
            return 1

        from bla import x, y
        ''')
        pkgs = packagefinder.get_packages(ast.parse(source).body)
        assert 'numpy' in pkgs
        assert 'torch' in pkgs
        assert 'ast' in pkgs
        assert 'padl' in pkgs
        assert 'requests' in pkgs
        assert 'ble' in pkgs
        assert 'bla' in pkgs


class TestDumpRequirements:
    def test_works(self):
        source = dedent('''
        import torch
        from ast import *
        ''')
        d = packagefinder.dump_requirements(ast.parse(source).body, strict=False)
        lines = d.splitlines()
        assert lines[0] == f'# created with python-{sys.version.split(" ")[0]}'
        assert lines[1].split('+')[0] == f'torch=={torch.__version__}'.split('+')[0]
        assert 'ast' not in d, 'builtin modules should not appear in requirements.'

    def test_strict_raises(self):
        source = dedent('''
        import padl
        ''')
        with raises(packagefinder.RequirementNotFound) as excinfo:
            packagefinder.dump_requirements(ast.parse(source).body, strict=True)
        assert excinfo.value.package == 'padl'
        assert str(excinfo.value) == 'Could not find an installed version of padl.'

    def test_not_strict_doesnt_raise(self):
        source = dedent('''
        import padl
        ''')
        with warns(UserWarning):
            packagefinder.dump_requirements(ast.parse(source).body, strict=False)
