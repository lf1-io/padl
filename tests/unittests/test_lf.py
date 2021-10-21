import pytest
import torch
from lf import lf
from collections import namedtuple
from lf.util_transforms import Batchify


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


def plus_one_(x):
    return x + 1


def get_info_(x):
    return x['info']


def test_context():
    plus_one = lf.trans(plus_one_)
    assert plus_one.lf_stage is None
    with plus_one.lf_set_stage('infer'):
        assert plus_one.lf_stage is 'infer'
        assert plus_one.lf_preprocess.lf_stage == 'infer'
        assert plus_one.lf_forward.lf_stage == 'infer'
        assert plus_one.lf_postprocess.lf_stage == 'infer'


def test_infer_apply():
    plus_one = lf.trans(plus_one_)
    assert plus_one.infer_apply(5) == 6


def test_eval_apply():
    plus_one = lf.trans(plus_one_)
    out = list(plus_one.eval_apply([5, 6], flatten=False))
    assert len(out) == 2
    assert out[0] == 6
    assert out[1] == 7

    get_info = lf.trans(get_info_)
    out = list(get_info.eval_apply([{'info': 'hello'}, {'info': 'dog'}], flatten=False))
    assert len(out) == 2
    assert out[0] == 'hello'
    assert out[1] == 'dog'


def test_compose():
    plus_one = lf.trans(plus_one_)
    comp1 = lf.Compose([plus_one, plus_one], module=None, stack=None)
    assert comp1(2) == 4
    assert comp1.infer_apply(2) == 4

    plus_one = lf.trans(plus_one_)
    comp2 = lf.Compose([plus_one, plus_one, Batchify()], module=None, stack=None)
    print(comp2.infer_apply(2))
    print(list(comp2.eval_apply([2, 2])))
    print(list(comp2.train_apply([2, 2])))


# TODO Add back once I can test Compose with preprocess step
# def test_loader_kwargs():
#     plus_one = lf.trans(plus_one_)
#     loader_kwargs = {'batch_size': 2}
#     out = list(plus_one.eval_apply([5, 6, 7, 8], loader_kwargs=loader_kwargs, flatten=False))
#     assert torch.all(out[0] == torch.tensor([6, 7]))
#     assert torch.all(out[1] == torch.tensor([8, 9]))
