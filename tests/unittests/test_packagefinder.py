import ast
from padl.dumptools import packagefinder


source_a = '''
from bla import x, y
import blu as z
import ble.lo

def x():
    return 1
'''


def test_getpackages():
    out = packagefinder.get_packages(ast.parse(source_a).body)
    assert out == {'bla', 'blu', 'ble'}
