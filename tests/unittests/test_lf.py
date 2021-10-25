import pytest
import torch
from lf import transform as lf, trans
from lf.util_transforms import Batchify, Unbatchify
from collections import namedtuple


@trans
def plus_one(x):
    return x + 1


@trans
def append_one(x):
    return x + "one"


@trans
def times_two(x):
    return x * 2


@trans
def plus(x, y):
    return x + y


@trans
def get_info(x):
    return x['info']


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
        request.cls.transform_2 = trans(simple_func) / trans(simple_func) / trans(simple_func)
        request.cls.transform_3 = plus_one / plus_one / trans(simple_func)

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

    def test_lf_preprocess(self):
        assert isinstance(self.transform_1.lf_preprocess, lf.Identity)

    def test_lf_forward(self):
        assert isinstance(self.transform_1.lf_forward, lf.Parallel)

    def test_lf_postprocess(self):
        assert isinstance(self.transform_1.lf_postprocess, lf.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply((2, 3, 4)) == (3, 6, 8)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([(2, 3, 4), (3, 3, 4)])) == [(3, 6, 8), (4, 6, 8)]

    def test_context(self):
        assert self.transform_1.lf_stage is None
        with self.transform_1.lf_set_stage('train'):
            assert self.transform_1.lf_stage is 'train'
            assert self.transform_1.lf_preprocess.lf_stage == 'train'
            assert self.transform_1.lf_forward.lf_stage == 'train'
            assert self.transform_1.lf_postprocess.lf_stage == 'train'


class TestRollout:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one + times_two + times_two
        request.cls.transform_2 = trans(simple_func) + trans(simple_func) + trans(simple_func)
        request.cls.transform_3 = plus_one + plus_one + trans(simple_func)

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

    def test_lf_preprocess(self):
        assert isinstance(self.transform_1.lf_preprocess, lf.Identity)

    def test_lf_forward(self):
        assert isinstance(self.transform_1.lf_forward, lf.Rollout)

    def test_lf_postprocess(self):
        assert isinstance(self.transform_1.lf_postprocess, lf.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(2) == (3, 4, 4)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([2, 3])) == [(3, 4, 4), (4, 6, 6)]

    def test_context(self):
        assert self.transform_1.lf_stage is None
        with self.transform_1.lf_set_stage('train'):
            assert self.transform_1.lf_stage is 'train'
            assert self.transform_1.lf_preprocess.lf_stage == 'train'
            assert self.transform_1.lf_forward.lf_stage == 'train'
            assert self.transform_1.lf_postprocess.lf_stage == 'train'


class TestCompose:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (plus_one >> times_two) >> times_two
        request.cls.transform_2 = plus_one >> (times_two >> times_two)
        request.cls.transform_3 = plus_one >> times_two >> times_two
        request.cls.transform_4 = plus_one >> plus_one >> plus_one
        request.cls.transform_5 = (
            plus_one
            >> Batchify()
            >> times_two
            >> times_two
            >> Unbatchify()
            >> plus_one
        )

    def test_associative(self):
        in_ = 123
        assert self.transform_1(in_) == self.transform_2(in_) == self.transform_3(in_)

    def test_output(self):
        assert self.transform_4(1) == 4

    def test_lf_preprocess(self):
        assert isinstance(self.transform_1.lf_preprocess, lf.Identity)
        assert isinstance(self.transform_5.lf_preprocess, lf.Compose)

    def test_lf_forward(self):
        assert isinstance(self.transform_1.lf_forward, lf.Compose)
        assert isinstance(self.transform_5.lf_forward, lf.Compose)

    def test_lf_postprocess(self):
        assert isinstance(self.transform_1.lf_postprocess, lf.Identity)
        assert isinstance(self.transform_5.lf_postprocess, lf.Compose)

    def test_infer_apply(self):
        assert self.transform_4.infer_apply(1) == 4
        assert self.transform_5.infer_apply(1) == torch.tensor(9)

    def test_eval_apply(self):
        assert list(self.transform_5.eval_apply([1, 1])) == [torch.tensor([9]), torch.tensor([9])]

    def test_train_apply(self):
        assert list(self.transform_5.eval_apply([1, 1])) == [torch.tensor([9]), torch.tensor([9])]

    def test_context(self):
        assert self.transform_1.lf_stage is None
        with self.transform_1.lf_set_stage('eval'):
            assert self.transform_1.lf_stage is 'eval'
            assert self.transform_1.lf_preprocess.lf_stage == 'eval'
            assert self.transform_1.lf_forward.lf_stage == 'eval'
            assert self.transform_1.lf_postprocess.lf_stage == 'eval'


class TestModel:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        transform_1 = (
            plus_one
            >> Batchify()
            >> times_two
            >> Unbatchify()
            >> plus_one
        )
        transform_2 = (
            plus_one
            >> Batchify()
            >> times_two
            >> Unbatchify()
            >> plus_one
        )
        request.cls.model_1 = (transform_1 / transform_2)
        request.cls.model_2 = (transform_1 + transform_2)

    def test_lf_preprocess(self):
        assert isinstance(self.model_1.lf_preprocess, lf.Parallel)
        assert isinstance(self.model_2.lf_preprocess, lf.Rollout)

    def test_lf_forward(self):
        assert isinstance(self.model_1.lf_forward, lf.Parallel)
        assert isinstance(self.model_2.lf_forward, lf.Parallel)

    def test_lf_postprocess(self):
        assert isinstance(self.model_1.lf_postprocess, lf.Parallel)
        assert isinstance(self.model_2.lf_postprocess, lf.Parallel)

    def test_infer_apply(self):
        # namedtuple(out_0=tensor(13), out_1=tensor(13))
        assert self.model_1.infer_apply((5, 5)) == (13, 13)
        # namedtuple(out_0=tensor(13), out_1=tensor(13))
        assert self.model_2.infer_apply(5) == (13, 13)

    def test_eval_apply(self):
        # [namedtuple(out_0=tensor([13]), out_1=tensor([13])),
        #  namedtuple(out_0=tensor([13]), out_1=tensor([13]))]
        assert list(self.model_1.eval_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        # FIXME Something wrong with rollout when using _lf_callyield
        assert list(self.model_2.eval_apply([5, 5])) == [(13, 13), (13, 13)]


class TestFunctionTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = get_info

    def test_lf_preprocess(self):
        assert isinstance(self.transform_1.lf_preprocess, lf.Identity)

    def test_lf_forward(self):
        assert isinstance(self.transform_1.lf_forward, lf.FunctionTransform)

    def test_lf_postprocess(self):
        assert isinstance(self.transform_1.lf_postprocess, lf.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(5) == 6

    def test_eval_apply(self):
        out = list(self.transform_1.eval_apply([5, 6], flatten=False))
        assert len(out) == 2
        assert out[0] == 6
        assert out[1] == 7

        out = list(self.transform_2.eval_apply([{'info': 'hello'}, {'info': 'dog'}], flatten=False))
        assert len(out) == 2
        assert out[0] == 'hello'
        assert out[1] == 'dog'

    def test_context(self):
        assert self.transform_1.lf_stage is None
        with self.transform_1.lf_set_stage('infer'):
            assert self.transform_1.lf_stage is 'infer'
            assert self.transform_1.lf_preprocess.lf_stage == 'infer'
            assert self.transform_1.lf_forward.lf_stage == 'infer'
            assert self.transform_1.lf_postprocess.lf_stage == 'infer'
