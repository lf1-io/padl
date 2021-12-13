import ast
import sys

from padl.dumptools import symfinder


def test__get_call_signature_a():
    source = 'a(x, y, z)'
    signature = symfinder._get_call_signature(source)
    assert signature == (['x', 'y', 'z'], {})


def test__get_call_signature_b():
    source = 'a(x, y, z=1)'
    signature = symfinder._get_call_signature(source)
    assert signature == (['x', 'y'], {'z': '1'})


def test__get_call_assignments_a():
    source = 'def f(x, y, z=1):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2, 3)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'x': '1', 'y': '2', 'z': '3'}


def test__get_call_assignments_b():
    source = 'def f(x, y, z=1):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'x': '1', 'y': '2', 'z': '1'}


def test__get_call_assignments_c():
    source = 'def f(*args, z=1):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2, 3, z=4)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'args': str([1, 2, 3]), 'z': '4'}


def test__get_call_assignments_d():
    source = 'def f(a, b, *args, z=1):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2, 3, z=4)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'a': '1', 'b': '2', 'args': '[3]', 'z': '4'}


def test__get_call_assignments_e():
    source = 'def f(a, b, *args, z=1, **kwargs):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2, 3, z=4, u=7, f=8)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'a': '1',
                           'b': '2',
                           'args': '[3]',
                           'z': '4',
                           'kwargs': "{'u': 7, 'f': 8}"}


def test__get_call_assignments_f():
    # only supported in python >= 3.8
    if sys.version_info.minor <= 7:
        return
    source = 'def f(a, b, /, c):pass'
    args = ast.parse(source).body[0].args
    values, keywords = symfinder._get_call_signature('f(1, 2, 3)')
    assignments = symfinder._get_call_assignments(args, source, values, keywords)
    assert assignments == {'a': '1',
                           'b': '2',
                           'c': '3'}
