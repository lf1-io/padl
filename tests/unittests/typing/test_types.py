from unittest import TestCase

from lf.typing.types import *
from lf.typing.patterns import val


class TestTypes(TestCase):
    def test_type_build(self):
        a = List()('x')
        b = Sequence()('x')
        a @ b

        self.assertEqual(val(b.type).name, 'List')

    def test_dict(self):
        fs = [
            (Any(), Any()),
            (Integer(), Any()),
            (List(len=10), List(len=20)),
            (List(len='?a'), List(len='?b')),
            (List(len='?a'), List(len='?a')),
            (either(List(), Integer()), List(len=123)),
            (List(len='?a'), (List(len='?a + 1'), List(len='?a - 1'))),
        ]
        for f in fs:
            a = Ftype.build(*f)
            b = Ftype.from_dict(a.to_dict())
            self.assertEqual(a, b)

    def test_typevar(self):
        a = List()('123')
        b = List()('123')
        a @ b
        a = List()('123')
        b = Sequence()('123')
        a @ b
        a = List(len='?a')('123')
        b = Sequence()('123')
        a @ b
        a = Tensor()('123')
        b = FloatTensor(shape=['?a', 1])('123')
        a @ b
