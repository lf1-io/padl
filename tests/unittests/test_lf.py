import pytest
import torch
from tadl import transforms as tadl, transform, Identity
from tadl.transforms import Batchify, Unbatchify
from collections import namedtuple
from tadl.exceptions import WrongDeviceError
from tests.fixtures.transforms import cleanup_checkpoint


@transform
def plus_one(x):
    return x + 1


@transform
def append_one(x):
    return x + "one"


@transform
def times_two(x):
    return x * 2


@transform
def plus(x, y):
    return x + y


@transform
def get_info(x):
    return x['info']


@transform
def complex_signature_func_1(a, b=10):
    return a+b


@transform
def complex_signature_func_2(*a, b= 10):
    return sum(a) + b


def simple_func(x):
    return x


class SimpleClass:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return x + self.a


@transform
class SimpleClassTransform:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return x + self.a


@transform
class ClassTransformWithManyArguments:
    def __init__(self, a, b, *args, c=1, d=2, **kwargs):
        self.a = a

    def __call__(self, x):
        return x + self.a


@transform
def trans_with_globals(x, y):
    return (plus >> times_two)(x, y)


@transform
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

    assert tadl._isinstance_of_namedtuple(namedtup_ins)
    assert not tadl._isinstance_of_namedtuple(tup)
    assert not tadl._isinstance_of_namedtuple(list(tup))
    assert not tadl._isinstance_of_namedtuple(1.)
    assert not tadl._isinstance_of_namedtuple('something')


class TestLFCallTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one >> (times_two + times_two)
        request.cls.transform_2 = transform(simple_func) + transform(simple_func) + transform(simple_func)
        request.cls.transform_3 = plus_one + times_two >> plus
        request.cls.transform_4 = plus_one + times_two >> complex_signature_func_1
        request.cls.transform_5 = plus_one >> complex_signature_func_1
        request.cls.transform_6 = plus_one + times_two >> complex_signature_func_2

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(1)
        assert self.transform_2.infer_apply(10)
        assert self.transform_3.infer_apply(1.4)
        assert self.transform_4.infer_apply(201)
        assert self.transform_5.infer_apply(11.1)
        assert self.transform_6.infer_apply(19)

    def test_save_load(self, cleanup_checkpoint):
        for transform_ in [self.transform_1,
                           self.transform_2,
                           self.transform_3,
                           self.transform_4,
                           self.transform_5,
                           self.transform_6]:
            transform_.td_save('test.tadl')
            t_ = tadl.load('test.tadl')
            assert t_.infer_apply(1)


class TestParallel:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one / times_two / times_two
        request.cls.transform_2 = transform(simple_func) / transform(simple_func) / transform(simple_func)
        request.cls.transform_3 = plus_one / plus_one / transform(simple_func)

    def test_output(self):
        in_ = (2, 2, 2)
        out = self.transform_1(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')

    def test_td_preprocess(self):
        assert isinstance(self.transform_1.td_preprocess, tadl.Identity)

    def test_td_forward(self):
        assert isinstance(self.transform_1.td_forward, tadl.Parallel)

    def test_td_postprocess(self):
        assert isinstance(self.transform_1.td_postprocess, tadl.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply((2, 3, 4)) == (3, 6, 8)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([(2, 3, 4), (3, 3, 4)])) == [(3, 6, 8), (4, 6, 8)]

    def test_context(self):
        assert self.transform_1.td_stage is None
        with self.transform_1.td_set_stage('train'):
            assert self.transform_1.td_stage is 'train'
            assert self.transform_1.td_preprocess.td_stage == 'train'
            assert self.transform_1.td_forward.td_stage == 'train'
            assert self.transform_1.td_postprocess.td_stage == 'train'

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        t1 = tadl.load('test.tadl')
        assert t1.infer_apply((2, 3, 4)) == (3, 6, 8)
        self.transform_2.td_save('test.tadl')
        _ = tadl.load('test.tadl')
        self.transform_3.td_save('test.tadl')
        _ = tadl.load('test.tadl')


class TestRollout:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one + times_two + times_two
        request.cls.transform_2 = transform(simple_func) + transform(simple_func) + transform(simple_func)
        request.cls.transform_3 = plus_one + plus_one + transform(simple_func)

    def test_output(self):
        in_ = 123
        out = self.transform_1(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert tadl._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')

    def test_td_preprocess(self):
        assert isinstance(self.transform_1.td_preprocess, tadl.Identity)

    def test_td_forward(self):
        assert isinstance(self.transform_1.td_forward, tadl.Rollout)

    def test_td_postprocess(self):
        assert isinstance(self.transform_1.td_postprocess, tadl.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(2) == (3, 4, 4)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([2, 3])) == [(3, 4, 4), (4, 6, 6)]

    def test_context(self):
        assert self.transform_1.td_stage is None
        with self.transform_1.td_set_stage('train'):
            assert self.transform_1.td_stage is 'train'
            assert self.transform_1.td_preprocess.td_stage == 'train'
            assert self.transform_1.td_forward.td_stage == 'train'
            assert self.transform_1.td_postprocess.td_stage == 'train'

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        t1 = tadl.load('test.tadl')
        assert t1.infer_apply(2) == (3, 4, 4)
        self.transform_2.td_save('test.tadl')
        _ = tadl.load('test.tadl')
        self.transform_3.td_save('test.tadl')
        _ = tadl.load('test.tadl')


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
            >> times_two - 'named_times_two'
            >> times_two
            >> Unbatchify()
        )

    def test_associative(self):
        in_ = 123
        assert self.transform_1(in_) == self.transform_2(in_) == self.transform_3(in_)

    def test_output(self):
        assert self.transform_4(1) == 4

    def test_td_preprocess(self):
        assert isinstance(self.transform_1.td_preprocess, tadl.Identity)
        assert isinstance(self.transform_5.td_preprocess, tadl.Compose)

    def test_td_forward(self):
        assert isinstance(self.transform_1.td_forward, tadl.Compose)
        assert isinstance(self.transform_5.td_forward, tadl.Compose)

    def test_td_postprocess(self):
        assert isinstance(self.transform_1.td_postprocess, tadl.Identity)
        assert isinstance(self.transform_5.td_postprocess, tadl.Unbatchify)

    def test_infer_apply(self):
        assert self.transform_4.infer_apply(1) == 4
        assert self.transform_5.infer_apply(1) == torch.tensor(8)

    def test_eval_apply(self):
        assert list(self.transform_5.eval_apply([1, 1])) == [torch.tensor([8]), torch.tensor([8])]

    def test_train_apply(self):
        # default
        assert list(self.transform_5.train_apply([1, 1])) == [torch.tensor([8]), torch.tensor([8])]
        # loader kwargs
        for batch in list(self.transform_5.train_apply(
            [1, 2, 1, 2],
            verbose=True,
            batch_size=2)
        ):
            assert torch.all(batch == torch.tensor([8, 12]))
        # flatten = True
        assert list(self.transform_5.train_apply(
            [1, 2, 1, 2],
            flatten=True,
            verbose=True,
            batch_size=2)
        ) == [torch.tensor([8]), torch.tensor([12]), torch.tensor([8]), torch.tensor([12])]

    def test_context(self):
        assert self.transform_1.td_stage is None
        with self.transform_1.td_set_stage('eval'):
            assert self.transform_1.td_stage is 'eval'
            assert self.transform_1.td_preprocess.td_stage == 'eval'
            assert self.transform_1.td_forward.td_stage == 'eval'
            assert self.transform_1.td_postprocess.td_stage == 'eval'

    def test_all_transforms_1(self):
        c = plus_one >> times_two >> times_two
        all_ = c.td_all_transforms()
        assert set(all_) == set([plus_one, times_two, c])

    def test_all_transforms_2(self):
        c = plus_one >> times_two >> trans_with_globals
        all_ = c.td_all_transforms()
        assert set(all_) == set([plus_one, times_two, c, trans_with_globals, plus])

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        _ = tadl.load('test.tadl')
        self.transform_2.td_save('test.tadl')
        _ = tadl.load('test.tadl')
        self.transform_3.td_save('test.tadl')
        _ = tadl.load('test.tadl')
        self.transform_4.td_save('test.tadl')
        t4 = tadl.load('test.tadl')
        assert t4.infer_apply(1) == 4
        self.transform_5.td_save('test.tadl')
        t5 = tadl.load('test.tadl')
        assert t5.infer_apply(1) == torch.tensor(8)

    def test_getitem(self):
        assert isinstance(self.transform_5[0], tadl.Transform)
        assert isinstance(self.transform_5[0:2], tadl.CompoundTransform)
        assert isinstance(self.transform_5[0:2], tadl.Compose)
        assert isinstance(self.transform_5['named_times_two'], tadl.Transform)
        with pytest.raises(ValueError):
            self.transform_5['other_name']
        with pytest.raises(TypeError):
            self.transform_5[2.1]


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
            >> Polynomial(1, 0) / Identity()
            >> Unbatchify()
            >> plus_one / times_two
        )

    def test_td_preprocess(self):
        assert isinstance(self.model_1.td_preprocess, tadl.Parallel)
        assert isinstance(self.model_2.td_preprocess, tadl.Rollout)
        assert isinstance(self.model_4.td_preprocess, tadl.Compose)

    def test_td_forward(self):
        assert isinstance(self.model_1.td_forward, tadl.Parallel)
        assert isinstance(self.model_2.td_forward, tadl.Parallel)

    def test_td_postprocess(self):
        assert isinstance(self.model_1.td_postprocess, tadl.Parallel)
        assert isinstance(self.model_2.td_postprocess, tadl.Parallel)
        assert isinstance(self.model_4.td_postprocess, tadl.Compose)

    def test_infer_apply(self):
        assert self.model_1.infer_apply((5, 5)) == (13, 13)
        assert self.model_2.infer_apply(torch.tensor(5)) == (13, 13)
        assert self.model_3.infer_apply(5) == (7, 20)
        assert self.model_4.infer_apply(5) == (8, 20)

    def test_eval_apply(self):
        assert list(self.model_1.eval_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_2.eval_apply(torch.tensor([5, 5]))) == [(13, 13), (13, 13)]
        assert list(self.model_3.eval_apply([5, 6])) == [(7, 20), (8, 24)]
        assert list(self.model_4.eval_apply([5, 6])) == [(8, 20), (9, 24)]

    def test_train_apply(self):
        assert list(self.model_1.train_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_2.train_apply(torch.tensor([5, 5]))) == [(13, 13), (13, 13)]
        assert list(self.model_3.train_apply([5, 6])) == [(7, 20), (8, 24)]
        assert list(self.model_4.train_apply([5, 6])) == [(8, 20), (9, 24)]

    def test_save_and_load(self, cleanup_checkpoint):
        #import pdb; pdb.set_trace()
        tadl.save(self.model_1, 'test.tadl')
        m1 = tadl.load('test.tadl')
        assert m1.infer_apply((5, 5)) == (13, 13)
        self.model_2.td_save('test.tadl')
        m2 = tadl.load('test.tadl')
        assert m2.infer_apply(5) == (13, 13)
        self.model_3.td_save('test.tadl')
        m3 = tadl.load('test.tadl')
        assert m3.infer_apply(5) == (7, 20)
        self.model_4.td_save('test.tadl')
        m4 = tadl.load('test.tadl') # TODO This Fails
        assert m4.infer_apply(5) == (8, 20)


class TestFunctionTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = get_info

    def test_td_preprocess(self):
        assert isinstance(self.transform_1.td_preprocess, tadl.Identity)

    def test_td_forward(self):
        assert isinstance(self.transform_1.td_forward, tadl.FunctionTransform)

    def test_td_postprocess(self):
        assert isinstance(self.transform_1.td_postprocess, tadl.Identity)

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
        assert self.transform_1.td_stage is None
        with self.transform_1.td_set_stage('infer'):
            assert self.transform_1.td_stage is 'infer'
            assert self.transform_1.td_preprocess.td_stage == 'infer'
            assert self.transform_1.td_forward.td_stage == 'infer'
            assert self.transform_1.td_postprocess.td_stage == 'infer'

    def test_all_transforms(self):
        all_ = trans_with_globals.td_all_transforms()
        assert set(all_) == set([plus, times_two, trans_with_globals])

    def test_td_to(self):
        self.transform_1.td_to('cpu')
        assert self.transform_1.td_device == 'cpu'

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        t1 = tadl.load('test.tadl')
        assert t1.infer_apply(5) == 6
        self.transform_2.td_save('test.tadl')
        _ = tadl.load('test.tadl')


def test_name():
    assert (plus_one - 'p1')._td_name == 'p1'
    assert plus_one._td_name is None


class TestTransformDeviceCheck:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (plus_one >> times_two) >> times_two
        request.cls.transform_2 = plus_one >> (times_two >> times_two)

    def test_device_check(self):
        self.transform_1.td_to('gpu')
        self.transform_1.transforms[1].td_to('cpu')

        with pytest.raises(WrongDeviceError):
            self.transform_1._td_forward_device_check()

        self.transform_2.td_to('gpu')
        assert self.transform_2._td_forward_device_check()


class TestClassTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = transform(SimpleClass)(2)
        request.cls.transform_2 = SimpleClassTransform(2)

    def test_infer_apply(self):
        self.transform_1.infer_apply(1) == 3
        self.transform_2.infer_apply(1) == 3

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        t1 = tadl.load('test.tadl')
        assert t1.infer_apply(1) == 3
        self.transform_2.td_save('test.tadl')
        t2 = tadl.load('test.tadl')
        assert t2.infer_apply(1) == 3

    def test_stored_arguments(self):
        c = ClassTransformWithManyArguments(1, 2, 3, 4, 5)


class TestTorchModuleTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = Polynomial(2, 3)

    def test_output(self):
        output = self.transform_1(1)
        assert output == 2

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(1)

    def test_device(self):
        self.transform_1.td_to('cpu')
        device = next(self.transform_1.td_layers[0].parameters()).device.type
        assert device == 'cpu'

    def test_td_layers(self):
        assert len(self.transform_1.td_layers) > 0

    def test_td_parameters(self):
        params = list(self.transform_1.td_parameters())
        assert len(params) == 2

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.td_save('test.tadl')
        t1 = tadl.load('test.tadl')
        assert t1.infer_apply(1) == 2


class TestLFImporter:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        from tadl.importer import numpy as inp
        request.cls.transform_1 = inp.sin
        request.cls.transform_2 = (inp.sin >> inp.sin)
        transform_temp = inp.sin
        request.cls.transform_3 = transform_temp + transform_temp >> inp.add
        request.cls.transform_4 = inp.cos + inp.cos >> inp.add

    def test_output(self):
        assert self.transform_1(2)
        assert self.transform_2(2.4)
        assert self.transform_3(1.1)
        assert self.transform_4(4.1)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(2)
        assert self.transform_2.infer_apply(2.4)
        assert self.transform_3.infer_apply(1.1)
        assert self.transform_4.infer_apply(4.1)

    """
    def test_save_load(self, cleanup_checkpoint):
        for transform_ in [self.transform_1,
                           self.transform_2,
                           self.transform_3,
                           self.transform_4,
                           ]:
            transform_.td_save('test.td')
            t_ = tadl.load('test.tadl')
            assert t_.infer_apply(1.3)
    """
