import pytest
import torch
from lf import transform as lf, trans, Identity
from lf.transform import Batchify, Unbatchify
from collections import namedtuple
from lf.exceptions import WrongDeviceError


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


@trans
def trans_with_globals(x, y):
    return (plus >> times_two)(x, y)


@trans
class Polynomial(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))
        self.b = torch.nn.Parameter(torch.tensor(float(b)))

    def forward(self, x):
        return x**self.a + x**self.b


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
        # default
        assert list(self.transform_5.train_apply([1, 1])) == [torch.tensor([9]), torch.tensor([9])]
        # loader kwargs
        for batch in list(self.transform_5.train_apply(
            [1, 2, 1, 2], loader_kwargs={'batch_size': 2})
        ):
            assert torch.all(batch == torch.tensor([9, 13]))
        # flatten = True
        assert list(self.transform_5.train_apply(
            [1, 2, 1, 2],
            loader_kwargs={'batch_size': 2},
            flatten=True)
        ) == [torch.tensor([9]), torch.tensor([13]), torch.tensor([9]), torch.tensor([13])]

    def test_context(self):
        assert self.transform_1.lf_stage is None
        with self.transform_1.lf_set_stage('eval'):
            assert self.transform_1.lf_stage is 'eval'
            assert self.transform_1.lf_preprocess.lf_stage == 'eval'
            assert self.transform_1.lf_forward.lf_stage == 'eval'
            assert self.transform_1.lf_postprocess.lf_stage == 'eval'

    def test_all_transforms_1(self):
        c = plus_one >> times_two >> times_two
        all_ = c.lf_all_transforms()
        assert set(all_) == set([plus_one, times_two, c])

    def test_all_transforms_2(self):
        c = plus_one >> times_two >> trans_with_globals
        all_ = c.lf_all_transforms()
        assert set(all_) == set([plus_one, times_two, c, trans_with_globals, plus])


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
        request.cls.model_3 = (
            plus_one + times_two
            >> plus_one / Batchify()
            >> Batchify() / times_two
        )
        request.cls.model_4 = (
            plus_one + times_two
            >> Batchify()
            # Testing if lf_forward is Identity() and not (Identity(), Identity())
            >> Identity() / Identity()
            >> Unbatchify()
            >> plus_one / times_two
        )

    def test_lf_preprocess(self):
        assert isinstance(self.model_1.lf_preprocess, lf.Parallel)
        assert isinstance(self.model_2.lf_preprocess, lf.Rollout)
        assert isinstance(self.model_4.lf_preprocess, lf.Compose)

    def test_lf_forward(self):
        assert isinstance(self.model_1.lf_forward, lf.Parallel)
        assert isinstance(self.model_2.lf_forward, lf.Parallel)

    def test_lf_postprocess(self):
        assert isinstance(self.model_1.lf_postprocess, lf.Parallel)
        assert isinstance(self.model_2.lf_postprocess, lf.Parallel)
        assert isinstance(self.model_4.lf_postprocess, lf.Compose)

    def test_infer_apply(self):
        assert self.model_1.infer_apply((5, 5)) == (13, 13)
        assert self.model_2.infer_apply(5) == (13, 13)
        assert self.model_3.infer_apply(5) == (7, 20)
        assert self.model_4.infer_apply(5) == (7, 20)

    def test_eval_apply(self):
        assert list(self.model_1.eval_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_2.eval_apply([5, 5])) == [(13, 13), (13, 13)]
        assert list(self.model_3.eval_apply([5, 6])) == [(7, 20), (8, 24)]
        assert list(self.model_4.eval_apply([5, 6])) == [(7, 20), (8, 24)]


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
        out = list(self.transform_1.eval_apply([5, 6]))
        assert len(out) == 2
        assert out[0] == 6
        assert out[1] == 7

        out = list(self.transform_2.eval_apply([{'info': 'hello'}, {'info': 'dog'}]))
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

    def test_all_transforms(self):
        all_ = trans_with_globals.lf_all_transforms()
        assert set(all_) == set([plus, times_two, trans_with_globals])

    def test_lf_to(self):
        self.transform_1.lf_to('cpu')
        assert self.transform_1.lf_device == 'cpu'


def test_name():
    assert (plus_one - 'p1')._lf_name == 'p1'
    assert plus_one._lf_name is None


class TestTransformDeviceCheck:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (plus_one >> times_two) >> times_two
        request.cls.transform_2 = plus_one >> (times_two >> times_two)

    def test_device_check(self):
        self.transform_1.lf_to('gpu')
        self.transform_1.transforms[1].lf_to('cpu')

        with pytest.raises(WrongDeviceError):
            self.transform_1._lf_forward_device_check()

        self.transform_2.lf_to('gpu')
        assert self.transform_2._lf_forward_device_check()


class TestTorchModuleTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = Polynomial(2, 3)

    def test_output(self):
        output = self.transform_1(1)
        assert output == 2

    def test_device(self):
        self.transform_1.lf_to('cpu')
        device = next(self.transform_1.lf_layers[0].parameters()).device.type
        assert device == 'cpu'

    def test_lf_layers(self):
        assert len(self.transform_1.lf_layers) > 0
