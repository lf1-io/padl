import pytest
import torch
from lf import lf
from collections import namedtuple


@lf.trans
def plus_one(x):
    return x + 1


@lf.trans
def append_one(x):
    return x + "one"


@lf.trans
def times_two(x):
    return x * 2


@lf.trans
def plus(x, y):
    return x + y


def simple_func(x):
    return x


def test_isinstance_of_namedtuple():
    tup = tuple([1, 2, 3])

    namedtup_type = namedtuple('something', 'a b c')
    namedtup_ins = namedtup_type(*tup)

    assert lf._isinstance_of_namedtuple(namedtup_ins)
    assert not lf._isinstance_of_namedtuple(tup)
    assert not lf._isinstance_of_namedtuple(list(tup))
    assert not lf._isinstance_of_namedtuple(1.)
    assert not lf._isinstance_of_namedtuple('something')


class TestParallel:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one / times_two / times_two
        request.cls.transform_2 = lf.trans(simple_func) / lf.trans(simple_func) / lf.trans(simple_func)
        request.cls.transform_3 = plus_one / plus_one / lf.trans(simple_func)

    def test_output(self):
        in_ = (2, 2, 2)
        out = self.transform_1(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')


class TestRollout:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one + times_two + times_two
        request.cls.transform_2 = lf.trans(simple_func) + lf.trans(simple_func) + lf.trans(simple_func)
        request.cls.transform_3 = plus_one + plus_one + lf.trans(simple_func)

    def test_output(self):
        in_ = 123
        out = self.transform_1(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert lf._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')


class TestCompose:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (plus_one >> times_two) >> times_two
        request.cls.transform_2 = plus_one >> (times_two >> times_two)
        request.cls.transform_3 = plus_one >> times_two >> times_two
        request.cls.transform_4 = plus_one >> plus_one >> plus_one

    def test_associative(self):
        in_ = 123
        assert self.transform_1(in_) == self.transform_2(in_) == self.transform_3(in_)

    def test_output(self):
        assert self.transform_4(1) == 4
