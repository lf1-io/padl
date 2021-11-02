from collections import OrderedDict
import pytest
import torch
from padl import transforms as pd, transform, Identity
from padl.transforms import Batchify, Unbatchify
from padl.dumptools.serialize import value
from collections import namedtuple
from padl.exceptions import WrongDeviceError


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
class ClassLookup:
    def __init__(self, dic):
        self.dic = dic

    def __call__(self, args):
        return [self.dic.get(x, len(self.dic)) for x in args]


@transform
class Polynomial(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))
        self.b = torch.nn.Parameter(torch.tensor(float(b)))

    def forward(self, x):
        return x**self.a + x**self.b


class PolynomialClass(torch.nn.Module):
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

    assert pd._isinstance_of_namedtuple(namedtup_ins)
    assert not pd._isinstance_of_namedtuple(tup)
    assert not pd._isinstance_of_namedtuple(list(tup))
    assert not pd._isinstance_of_namedtuple(1.)
    assert not pd._isinstance_of_namedtuple('something')


class TestLFCallTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one >> (times_two + times_two)
        request.cls.transform_2 = transform(simple_func) + transform(simple_func) \
            + transform(simple_func)
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

    def test_save_load(self, tmp_path):
        for transform_ in [self.transform_1,
                           self.transform_2,
                           self.transform_3,
                           self.transform_4,
                           self.transform_5,
                           self.transform_6]:
            transform_.pd_save(tmp_path / 'test.padl', True)
            t_ = pd.load(tmp_path / 'test.padl')
            assert t_.infer_apply(1)


class TestMap:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = ~plus_one
        request.cls.transform_2 = transform(simple_func) / ~plus_one
        request.cls.transform_3 = ~times_two + ~plus_one
        request.cls.transform_4 = transform(lambda x: [x, x, x]) >> ~plus_one

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)
        assert isinstance(self.transform_2.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Map)
        assert isinstance(self.transform_2.pd_forward, pd.Parallel)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_2.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply([2, 3, 4]) == [3, 4, 5]
        assert self.transform_2.infer_apply((1, [2, 3, 4])) == (1, [3, 4, 5])
        assert self.transform_3.infer_apply([2, 3, 4]) == ([4, 6, 8], [3, 4, 5])
        assert self.transform_4.infer_apply(1) == [2, 2, 2]

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([[2, 3], [3, 4]])) == [[3, 4], [4, 5]]
        assert list(self.transform_2.eval_apply(([1, [2, 3]], (2, [3, 4])))) == [(1, [3, 4]),
                                                                                 (2, [4, 5])]
        assert list(self.transform_3.eval_apply([[2, 3], [2, 3]])) == \
               [([4, 6], [3, 4]), ([4, 6], [3, 4])]
        assert list(self.transform_4.eval_apply([1])) == [[2, 2, 2]]

    def test_train_apply(self):
        assert list(self.transform_1.train_apply([[2, 3], [3, 4]])) == [[3, 4], [4, 5]]
        assert list(self.transform_2.train_apply(([1, [2, 3]], (2, [3, 4])))) == [(1, [3, 4]),
                                                                                  (2, [4, 5])]
        assert list(self.transform_3.train_apply([[2, 3], [2, 3]])) == \
               [([4, 6], [3, 4]), ([4, 6], [3, 4])]
        assert list(self.transform_4.train_apply([1])) == [[2, 2, 2]]

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply([2, 3, 4]) == [3, 4, 5]
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        t2 = pd.load(tmp_path / 'test.padl')
        assert t2.infer_apply((1, [2, 3, 4])) == (1, [3, 4, 5])
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        t3 = pd.load(tmp_path / 'test.padl')
        assert t3.infer_apply([2, 3, 4]) == ([4, 6, 8], [3, 4, 5])
        self.transform_4.pd_save(tmp_path / 'test.padl', True)
        t4 = pd.load(tmp_path / 'test.padl')
        assert t4.infer_apply(1) == [2, 2, 2]


class TestParallel:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one / times_two / times_two
        request.cls.transform_2 = transform(simple_func) / transform(simple_func) / transform(simple_func)
        request.cls.transform_3 = plus_one / plus_one / transform(simple_func)

    def test_output(self):
        in_ = (2, 2, 2)
        out = self.transform_1(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Parallel)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply((2, 3, 4)) == (3, 6, 8)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([(2, 3, 4), (3, 3, 4)])) == [(3, 6, 8), (4, 6, 8)]

    def test_context(self):
        assert self.transform_1.pd_stage is None
        with self.transform_1.pd_set_stage('train'):
            assert self.transform_1.pd_stage is 'train'
            assert self.transform_1.pd_preprocess.pd_stage == 'train'
            assert self.transform_1.pd_forward.pd_stage == 'train'
            assert self.transform_1.pd_postprocess.pd_stage == 'train'

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply((2, 3, 4)) == (3, 6, 8)
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')


class TestRollout:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one + times_two + times_two
        request.cls.transform_2 = transform(simple_func) + transform(simple_func) + transform(simple_func)
        request.cls.transform_3 = plus_one + plus_one + transform(simple_func)

    def test_output(self):
        in_ = 123
        out = self.transform_1(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one', 'times_two_0', 'times_two_1')

        out = self.transform_2(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('out_0', 'out_1', 'out_2')

        out = self.transform_3(in_)
        assert pd._isinstance_of_namedtuple(out)
        assert out._fields == ('plus_one_0', 'plus_one_1', 'out_2')

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Rollout)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(2) == (3, 4, 4)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([2, 3])) == [(3, 4, 4), (4, 6, 6)]

    def test_context(self):
        assert self.transform_1.pd_stage is None
        with self.transform_1.pd_set_stage('train'):
            assert self.transform_1.pd_stage is 'train'
            assert self.transform_1.pd_preprocess.pd_stage == 'train'
            assert self.transform_1.pd_forward.pd_stage == 'train'
            assert self.transform_1.pd_postprocess.pd_stage == 'train'

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(2) == (3, 4, 4)
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')


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

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)
        assert isinstance(self.transform_5.pd_preprocess, pd.Compose)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Compose)
        assert isinstance(self.transform_5.pd_forward, pd.Compose)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_5.pd_postprocess, pd.Unbatchify)

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
        assert self.transform_1.pd_stage is None
        with self.transform_1.pd_set_stage('eval'):
            assert self.transform_1.pd_stage is 'eval'
            assert self.transform_1.pd_preprocess.pd_stage == 'eval'
            assert self.transform_1.pd_forward.pd_stage == 'eval'
            assert self.transform_1.pd_postprocess.pd_stage == 'eval'

    def test_all_transforms_1(self):
        c = plus_one >> times_two >> times_two
        all_ = c._pd_all_transforms()
        assert set(all_) == set([plus_one, times_two, c])

    def test_all_transforms_2(self):
        c = plus_one >> times_two >> trans_with_globals
        all_ = c._pd_all_transforms()
        assert set(all_) == set([plus_one, times_two, c, trans_with_globals, plus])

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        _ = pd.load(tmp_path / 'test.padl')
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')
        self.transform_4.pd_save(tmp_path / 'test.padl', True)
        t4 = pd.load(tmp_path / 'test.padl')
        assert t4.infer_apply(1) == 4
        self.transform_5.pd_save(tmp_path / 'test.padl', True)
        t5 = pd.load(tmp_path / 'test.padl')
        assert t5.infer_apply(1) == torch.tensor(8)

    def test_getitem(self):
        assert isinstance(self.transform_5[0], pd.Transform)
        assert isinstance(self.transform_5[0:2], pd.CompoundTransform)
        assert isinstance(self.transform_5[0:2], pd.Compose)
        assert isinstance(self.transform_5['named_times_two'], pd.Transform)
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
        ) - 'transform_2'
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
        ) - 'model_4'

    def test_pd_preprocess(self):
        assert isinstance(self.model_1.pd_preprocess, pd.Parallel)
        assert isinstance(self.model_2.pd_preprocess, pd.Rollout)
        assert isinstance(self.model_4.pd_preprocess, pd.Compose)

    def test_pd_forward(self):
        assert isinstance(self.model_1.pd_forward, pd.Parallel)
        assert isinstance(self.model_2.pd_forward, pd.Parallel)

    def test_pd_postprocess(self):
        assert isinstance(self.model_1.pd_postprocess, pd.Parallel)
        assert isinstance(self.model_2.pd_postprocess, pd.Parallel)
        assert isinstance(self.model_4.pd_postprocess, pd.Compose)

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

    def test_save_and_load(self, tmp_path):
        pd.save(self.model_1, tmp_path / 'test.padl')
        m1 = pd.load(tmp_path / 'test.padl')
        assert m1.infer_apply((5, 5)) == (13, 13)
        self.model_2.pd_save(tmp_path / 'test.padl', True)
        m2 = pd.load(tmp_path / 'test.padl')
        assert m2.infer_apply(5) == (13, 13)
        self.model_3.pd_save(tmp_path / 'test.padl', True)
        m3 = pd.load(tmp_path / 'test.padl')
        assert m3.infer_apply(5) == (7, 20)
        self.model_4.pd_save(tmp_path / 'test.padl', True)
        m4 = pd.load(tmp_path / 'test.padl')
        assert m4.infer_apply(5) == (8, 20)


class TestFunctionTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = get_info

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.FunctionTransform)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)

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
        assert self.transform_1.pd_stage is None
        with self.transform_1.pd_set_stage('infer'):
            assert self.transform_1.pd_stage is 'infer'
            assert self.transform_1.pd_preprocess.pd_stage == 'infer'
            assert self.transform_1.pd_forward.pd_stage == 'infer'
            assert self.transform_1.pd_postprocess.pd_stage == 'infer'

    def test_all_transforms(self):
        all_ = trans_with_globals._pd_all_transforms()
        assert set(all_) == set([plus, times_two, trans_with_globals])

    def test_pd_to(self):
        self.transform_1.pd_to('cpu')
        assert self.transform_1.pd_device == 'cpu'

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl', True)
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(5) == 6
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        _ = pd.load(tmp_path / 'test.padl')


def test_name():
    assert (plus_one - 'p1')._pd_name == 'p1'
    assert plus_one._pd_name is None


class TestTransformDeviceCheck:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (plus_one >> times_two) >> times_two
        request.cls.transform_2 = plus_one >> (times_two >> times_two)

    def test_device_check(self):
        self.transform_1.pd_to('gpu')
        self.transform_1.transforms[1].pd_to('cpu')

        with pytest.raises(WrongDeviceError):
            self.transform_1._pd_forward_device_check()

        self.transform_2.pd_to('gpu')
        assert self.transform_2._pd_forward_device_check()


class TestClassTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = transform(SimpleClass)(2)
        request.cls.transform_2 = SimpleClassTransform(2)
        dic = {s: i for i, s in enumerate('abcdefghijklmnop')}
        request.cls.transform_3 = ClassLookup(dic=value(dic))
        request.cls.dic = dic

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(1) == 3
        assert self.transform_2.infer_apply(1) == 3
        assert self.transform_3.infer_apply('abc') == [0, 1, 2]

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(1) == 3
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        t2 = pd.load(tmp_path / 'test.padl')
        assert t2.infer_apply(1) == 3
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        t3 = pd.load(tmp_path / 'test.padl')
        assert t3.dic == self.dic
        assert t3.infer_apply('abc') == [0, 1, 2]

    def test_stored_arguments(self):
        c = ClassTransformWithManyArguments(1, 2, 3, 4, 5)
        assert c._pd_arguments == OrderedDict([('a', 1), ('b', 2), ('args', (3, 4, 5))])


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
        self.transform_1.pd_to('cpu')
        device = next(self.transform_1.pd_layers[0].parameters()).device.type
        assert device == 'cpu'

    def test_pd_layers(self):
        assert len(self.transform_1.pd_layers) > 0

    def test_pd_parameters(self):
        params = list(self.transform_1.pd_parameters())
        assert len(params) == 2

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(1) == 2


class TestClassInstance:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = transform(SimpleClass(1))
        request.cls.transform_2 = transform(PolynomialClass(1, 2))

    def test_wrap(self):
        assert isinstance(self.transform_1, SimpleClass)
        assert isinstance(self.transform_1, pd.Transform)
        assert isinstance(self.transform_2, PolynomialClass)
        assert isinstance(self.transform_2, pd.Transform)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(1) == 2
        assert self.transform_2.infer_apply(1) == 2

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([1])) == [2]
        assert list(self.transform_2.eval_apply([2])) == [6]

    def test_train_apply(self):
        assert list(self.transform_1.train_apply([1])) == [2]
        assert list(self.transform_2.train_apply([2])) == [6]

    def test_print(self):
        assert str(self.transform_1)
        assert str(self.transform_2)

    def test_pd_layers(self):
        assert len(self.transform_2.pd_layers) > 0

    def test_pd_parameters(self):
        params = list(self.transform_2.pd_parameters())
        assert len(params) == 2

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(1) == 2
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        t2 = pd.load(tmp_path / 'test.padl')
        assert t2.infer_apply(1) == 2
