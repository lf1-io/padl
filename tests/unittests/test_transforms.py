import os
from collections import OrderedDict
import pytest
import torch
from padl import transforms as pd, transform, Identity, batch, unbatch, group
from padl.transforms import Batchify, Unbatchify, TorchModuleTransform
from padl.dumptools.serialize import value
import padl
from collections import namedtuple
from padl.transforms import Transform

GLOBAL_1 = 0
GLOBAL_1 = GLOBAL_1 + 5


class PrettyMock:
    @staticmethod
    def text(x):
        return x


@transform
def plus_global(x):
    return x + GLOBAL_1


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
def complex_signature_func_2(*a, b=10):
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
    def __init__(self, a, b, pd_save_options=None):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))
        self.b = torch.nn.Parameter(torch.tensor(float(b)))
        if pd_save_options is not None:
            self.pd_save_options = pd_save_options

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


class TestNamedTupleOutput:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = request.cls.transform_1 >> (times_two + times_two)
        request.cls.transform_3 = request.cls.transform_2 >> (times_two / times_two)

    def test_call(self):
        assert not pd._isinstance_of_namedtuple(self.transform_1(1))
        assert pd._isinstance_of_namedtuple(self.transform_2(1))
        assert pd._isinstance_of_namedtuple(self.transform_3(1))

    def test_infer_apply(self):
        assert not pd._isinstance_of_namedtuple(self.transform_1.infer_apply(1))
        assert pd._isinstance_of_namedtuple(self.transform_2.infer_apply(1))
        assert pd._isinstance_of_namedtuple(self.transform_3.infer_apply(1))

    def test_eval_apply(self):
        assert not any(list(map(pd._isinstance_of_namedtuple,
                                self.transform_1.eval_apply([1, 2, 3]))))
        assert all(list(map(pd._isinstance_of_namedtuple,
                            self.transform_2.eval_apply([1, 2, 3]))))
        assert all(list(map(pd._isinstance_of_namedtuple,
                            self.transform_3.eval_apply([1, 2, 3]))))

    def test_train_apply(self):
        assert not any(list(map(pd._isinstance_of_namedtuple,
                                self.transform_1.train_apply([1, 2, 3]))))
        assert all(list(map(pd._isinstance_of_namedtuple,
                            self.transform_2.train_apply([1, 2, 3]))))
        assert all(list(map(pd._isinstance_of_namedtuple,
                            self.transform_3.train_apply([1, 2, 3]))))


def test__pd_process_options():
    @transform
    class A:
        def __call__(self, x):
            return x

    class B(A):
        ...

    @transform
    class C:
        def __call__(self, x):
            return x

    options = {A: 1, B: 2, C: 3}
    assert A()._pd_process_options(options) == 1
    assert B()._pd_process_options(options) == 2
    assert C()._pd_process_options(options) == 3


class TestPADLCallTransform:
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

    def test_pprintt(self):
        self.transform_1._repr_pretty_(PrettyMock, False)
        self.transform_2._repr_pretty_(PrettyMock, False)
        self.transform_3._repr_pretty_(PrettyMock, False)
        self.transform_4._repr_pretty_(PrettyMock, False)
        self.transform_5._repr_pretty_(PrettyMock, False)
        self.transform_6._repr_pretty_(PrettyMock, False)

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
        request.cls.transform_5 = Batchify() >> ~plus_one
        request.cls.transform_6 = (
            Batchify() / Identity()
            >> ~plus_one
        )
        request.cls.transform_7 = (
                Batchify() / Identity()
                >> ~plus_one
                >> Unbatchify() / Identity()
        )

    def test_pprintt(self):
        self.transform_1._repr_pretty_(PrettyMock, False)
        self.transform_2._repr_pretty_(PrettyMock, False)
        self.transform_3._repr_pretty_(PrettyMock, False)
        self.transform_4._repr_pretty_(PrettyMock, False)
        self.transform_5._repr_pretty_(PrettyMock, False)
        self.transform_6._repr_pretty_(PrettyMock, False)

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)
        assert isinstance(self.transform_2.pd_preprocess, pd.Identity)
        assert isinstance(self.transform_5.pd_preprocess, pd.Batchify)
        assert isinstance(self.transform_6.pd_preprocess, pd.Compose)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Map)
        assert isinstance(self.transform_2.pd_forward, pd.Parallel)
        assert isinstance(self.transform_5.pd_forward, pd.Map)
        assert isinstance(self.transform_6.pd_forward, pd.Parallel)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_2.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_5.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_6.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_7.pd_postprocess, pd.Compose)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply([2, 3, 4]) == (3, 4, 5)
        assert self.transform_2.infer_apply((1, [2, 3, 4])) == (1, (3, 4, 5))
        assert self.transform_3.infer_apply([2, 3, 4]) == ((4, 6, 8), (3, 4, 5))
        assert self.transform_4.infer_apply(1) == (2, 2, 2)
        assert self.transform_5.infer_apply([1, 1, 1]) == (2, 2, 2)
        assert self.transform_6.infer_apply([1, 2]) == (2, 3)

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([[2, 3], [3, 4]])) == [(3, 4), (4, 5)]
        assert list(self.transform_2.eval_apply(([1, [2, 3]], (2, [3, 4])))) == [(1, (3, 4)),
                                                                                 (2, (4, 5))]
        assert list(self.transform_3.eval_apply([[2, 3], [2, 3]])) == \
               [((4, 6), (3, 4)), ((4, 6), (3, 4))]
        assert list(self.transform_4.eval_apply([1])) == [(2, 2, 2)]

    def test_train_apply(self):
        assert list(self.transform_1.train_apply([[2, 3], [3, 4]])) == [(3, 4), (4, 5)]
        assert list(self.transform_2.train_apply(([1, [2, 3]], (2, [3, 4])))) == [(1, (3, 4)),
                                                                                  (2, (4, 5))]
        assert list(self.transform_3.eval_apply([[2, 3], [2, 3]])) == \
               [((4, 6), (3, 4)), ((4, 6), (3, 4))]
        assert list(self.transform_4.train_apply([1])) == [(2, 2, 2)]

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply([2, 3, 4]) == (3, 4, 5)
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        t2 = pd.load(tmp_path / 'test.padl')
        assert t2.infer_apply((1, [2, 3, 4])) == (1, (3, 4, 5))
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        t3 = pd.load(tmp_path / 'test.padl')
        assert t3.infer_apply([2, 3, 4]) == ((4, 6, 8), (3, 4, 5))
        self.transform_4.pd_save(tmp_path / 'test.padl', True)
        t4 = pd.load(tmp_path / 'test.padl')
        assert t4.infer_apply(1) == (2, 2, 2)


class TestParallel:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one / times_two / times_two
        request.cls.transform_2 = transform(simple_func) / transform(simple_func) / transform(simple_func)
        request.cls.transform_3 = plus_one / plus_one / transform(simple_func)
        request.cls.transform_4 = (
            plus_one / plus_one
            >> transform(lambda x: x[0] * x[1])
        )

    def test_pprintt(self):
        self.transform_1._repr_pretty_(PrettyMock, False)
        self.transform_2._repr_pretty_(PrettyMock, False)
        self.transform_3._repr_pretty_(PrettyMock, False)
        self.transform_4._repr_pretty_(PrettyMock, False)

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
        assert isinstance(self.transform_4.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Parallel)
        assert isinstance(self.transform_4.pd_forward, pd.Compose)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_4.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply((2, 3, 4)) == (3, 6, 8)
        assert self.transform_4.infer_apply((2, 4)) == 15

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([(2, 3, 4), (3, 3, 4)])) == [(3, 6, 8), (4, 6, 8)]

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
        request.cls.transform_4 = (
            (Batchify() >> plus_one) + (times_two >> Batchify())
        )
        request.cls.transform_5 = (
            (times_two >> Batchify()) + (Batchify() >> plus_one)
        )
        request.cls.transform_6 = (
            plus_one / times_two
            >> times_two + times_two
        )

    def test_pprintt(self):
        self.transform_1._repr_pretty_(PrettyMock, False)
        self.transform_2._repr_pretty_(PrettyMock, False)
        self.transform_3._repr_pretty_(PrettyMock, False)
        self.transform_4._repr_pretty_(PrettyMock, False)
        self.transform_5._repr_pretty_(PrettyMock, False)
        self.transform_6._repr_pretty_(PrettyMock, False)

    def test_identity_split(self):
        new_iden = Identity() - 'new_name'
        test = (
            plus_one
            >> batch
            >> new_iden + new_iden
            >> plus
            >> unbatch
            >> new_iden + new_iden
            >> plus
        )
        assert str(test.pd_forward) == str(new_iden + new_iden >> plus)

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
        assert isinstance(self.transform_6.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.Rollout)
        assert isinstance(self.transform_6.pd_forward, pd.Compose)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)
        assert isinstance(self.transform_6.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(2) == (3, 4, 4)
        assert self.transform_4.infer_apply(2) == (3, 4)
        assert self.transform_5.infer_apply(2) == (4, 3)
        assert self.transform_6.infer_apply((2, 2)) == ((3, 4, 3, 4), (3, 4, 3, 4))

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([2, 3])) == [(3, 4, 4), (4, 6, 6)]

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
        request.cls.transform_6 = (
            times_two
            >> plus_one + plus_one
        )
        request.cls.transform_7 = (
                times_two
                >> plus_one + plus_one
                >> Identity() / Unbatchify()
        )
        request.cls.transform_8 = (
                times_two
                >> Unbatchify()
        )
        request.cls.transform_9 = (
                times_two
                >> Batchify()
                >> Unbatchify()
        )
        request.cls.transform_10 = (
            times_two
            >> Batchify()
            >> Unbatchify()
            >> Unbatchify()
        )

    def test_unbatchify_position(self):
        assert self.transform_7.infer_apply(1) == (3, 3)
        assert self.transform_8.infer_apply(1) == 2
        assert self.transform_9.infer_apply(1) == 2
        with pytest.raises(AssertionError):
            self.transform_10.infer_apply(1)

    def test_pprintt(self):
        self.transform_1._repr_pretty_(PrettyMock, False)
        self.transform_2._repr_pretty_(PrettyMock, False)
        self.transform_3._repr_pretty_(PrettyMock, False)
        self.transform_4._repr_pretty_(PrettyMock, False)
        self.transform_5._repr_pretty_(PrettyMock, False)

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
        assert list(self.transform_5.eval_apply([1, 1])) == [torch.tensor(8), torch.tensor(8)]

    def test_train_apply(self):
        # default
        assert list(self.transform_5.train_apply([1, 1])) == [torch.tensor(8), torch.tensor(8)]
        # loader kwargs
        for out in list(self.transform_5.train_apply(
            [1, 1, 1, 1],
            batch_size=2)
        ):
            assert out == torch.tensor(8)
        assert list(self.transform_5.train_apply(
            [1, 2, 1, 2],
            flatten=True,
            batch_size=2)
        ) == [torch.tensor(8), torch.tensor(12), torch.tensor(8), torch.tensor(12)]

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
        assert isinstance(self.transform_5[0:2], pd.Pipeline)
        assert isinstance(self.transform_5[0:2], pd.Compose)
        assert isinstance(self.transform_5['named_times_two'], pd.Transform)
        with pytest.raises(ValueError):
            self.transform_5['other_name']
        with pytest.raises(TypeError):
            self.transform_5[2.1]
        assert isinstance(self.transform_6['plus_one'], pd.Transform)


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
        request.cls.model_5 = ~transform_1
        request.cls.model_6 = (
            Batchify()
            >> plus_one
            >> plus_one + times_two
        )
        request.cls.model_7 = (
            plus_one + times_two
            >> plus_one / Batchify()
            >> plus_one / plus_one
            >> Batchify() / times_two
            >> Unbatchify()
            >> plus_one / plus_one
        )

    @pytest.fixture(scope='class')
    def to_tensor(self):
        return transform(lambda x: torch.tensor(x))

    @pytest.fixture(scope='class')
    def lin(self):
        return transform(torch.nn.Linear(2, 2))

    @pytest.fixture(scope='class')
    def post(self):
        return transform(lambda x: x.sum(-1).topk(1, -1).indices.item())

    def test_pprintt(self):
        self.model_1._repr_pretty_(PrettyMock, False)
        self.model_2._repr_pretty_(PrettyMock, False)
        self.model_3._repr_pretty_(PrettyMock, False)
        self.model_4._repr_pretty_(PrettyMock, False)
        self.model_5._repr_pretty_(PrettyMock, False)
        self.model_6._repr_pretty_(PrettyMock, False)

    def test_pd_preprocess(self):
        assert isinstance(self.model_1.pd_preprocess, pd.Parallel)
        assert isinstance(self.model_2.pd_preprocess, pd.Rollout)
        assert isinstance(self.model_4.pd_preprocess, pd.Compose)
        assert isinstance(self.model_5.pd_preprocess, pd.Map)
        assert isinstance(self.model_6.pd_preprocess, pd.Batchify)
        assert isinstance(self.model_7.pd_preprocess, pd.Compose)

    def test_pd_forward(self):
        assert isinstance(self.model_1.pd_forward, pd.Parallel)
        assert isinstance(self.model_2.pd_forward, pd.Parallel)
        assert isinstance(self.model_5.pd_forward, pd.Map)
        assert isinstance(self.model_6.pd_forward, pd.Compose)
        assert isinstance(self.model_7.pd_forward, pd.Compose)

    def test_pd_postprocess(self):
        assert isinstance(self.model_1.pd_postprocess, pd.Parallel)
        assert isinstance(self.model_2.pd_postprocess, pd.Parallel)
        assert isinstance(self.model_4.pd_postprocess, pd.Compose)
        assert isinstance(self.model_5.pd_postprocess, pd.Map)
        assert isinstance(self.model_6.pd_postprocess, pd.Identity)
        assert isinstance(self.model_7.pd_postprocess, pd.Compose)

    def test_infer_apply(self):
        assert self.model_1.infer_apply((5, 5)) == (13, 13)
        assert self.model_2.infer_apply(torch.tensor(5)) == (13, 13)
        assert self.model_3.infer_apply(5) == (7, 20)
        assert self.model_4.infer_apply(5) == (8, 20)
        assert self.model_5.infer_apply((5, 5)) == (13, 13)
        assert self.model_6.infer_apply(5) == (7, 12)
        assert self.model_7.infer_apply(5) == (9, 23)

    def test_eval_apply(self):
        assert list(self.model_1.eval_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_2.eval_apply(torch.tensor([5, 5]))) == [(13, 13), (13, 13)]
        assert list(self.model_3.eval_apply([5, 6])) == [(7, 20), (8, 24)]
        assert list(self.model_4.eval_apply([5, 6])) == [(8, 20), (9, 24)]
        assert list(self.model_5.eval_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_6.eval_apply([5, 6])) == [(7, 12), (8, 14)]
        assert list(self.model_7.eval_apply([5, 6])) == [(9, 23), (10, 27)]

    def test_train_apply(self):
        assert list(self.model_1.train_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]
        assert list(self.model_2.train_apply(torch.tensor([5, 5]))) == [(13, 13), (13, 13)]
        assert list(self.model_3.train_apply([5, 6])) == [(7, 20), (8, 24)]
        assert list(self.model_4.train_apply([5, 6])) == [(8, 20), (9, 24)]
        assert list(self.model_5.train_apply([(5, 5), (5, 5)])) == [(13, 13), (13, 13)]

    def test_save_and_load(self, tmp_path):
        pd.save(self.model_1, tmp_path / 'test.padl', compress=True, force_overwrite=True)
        m1 = pd.load(tmp_path / 'test.padl')
        assert m1.infer_apply((5, 5)) == (13, 13)
        pd.save(self.model_2, tmp_path / 'test.padl', compress=True, force_overwrite=True)
        m2 = pd.load(tmp_path / 'test.padl')
        assert m2.infer_apply(5) == (13, 13)
        pd.save(self.model_3, tmp_path / 'test1.padl', force_overwrite=True)
        m3 = pd.load(tmp_path / 'test1.padl')
        assert m3.infer_apply(5) == (7, 20)
        self.model_4.pd_save(tmp_path / 'test1.padl', force_overwrite=True)
        m4 = pd.load(tmp_path / 'test1.padl')
        assert m4.infer_apply(5) == (8, 20)
        self.model_5.pd_save(tmp_path / 'test1.padl', force_overwrite=True)
        m5 = pd.load(tmp_path / 'test1.padl')
        assert m5.infer_apply((5, 5)) == (13, 13)

    def test_pd_splits_compose(self, to_tensor, lin, post):
        t = (to_tensor >> batch
             >> (lin >> lin) + (lin >> lin) + (lin >> lin)
             >> (unbatch >> post) / (unbatch >> post) / (unbatch >> post)
        )
        t_preprocess = to_tensor >> batch
        t_forward = group((lin >> lin) + (lin >> lin) + (lin >> lin))
        t_postprocess = group((unbatch >> post) / (unbatch >> post) / (unbatch >> post))
        assert str(t.pd_preprocess) == str(t_preprocess)
        assert str(t.pd_forward) == str(t_forward)
        assert str(t.pd_postprocess) == str(t_postprocess)

    def test_pd_splits_compose_with_group(self, to_tensor, lin, post):
        t = (to_tensor >> batch
             >> group((lin >> lin) + (lin >> lin)) + (lin >> lin)
             >> group((unbatch >> post) / (unbatch >> post)) / (unbatch >> post)
        )
        t_preprocess = to_tensor >> batch
        t_forward = group(group((lin >> lin) + (lin >> lin)) + (lin >> lin))
        t_postprocess = group(group((unbatch >> post) / (unbatch >> post)) / (unbatch >> post))
        assert str(t.pd_preprocess) == str(t_preprocess)
        assert str(t.pd_forward) == str(t_forward)
        assert str(t.pd_postprocess) == str(t_postprocess)

    def test_pd_splits_parallel(self, to_tensor, lin):
        t = (((to_tensor >> batch >> lin) + (to_tensor >> batch >> lin))
             / ((to_tensor >> batch) + (to_tensor >> batch))
        ) - 'name'
        g = t / Identity()
        t_preprocess = group(group((to_tensor >> batch) + (to_tensor >> batch)) /
                             group((to_tensor >> batch) + (to_tensor >> batch)))
        g_preprocess = group((t_preprocess - 'name') / Identity())
        t_forward = group(group(lin / lin) / Identity())
        g_forward = group((t_forward - 'name') / Identity())
        assert str(t.pd_preprocess) == str(t_preprocess)
        assert str(g.pd_preprocess) == str(g_preprocess)
        assert str(t.pd_forward) == str(t_forward)
        assert str(g.pd_forward) == str(g_forward)

    def test_pd_splits_rollout(self, to_tensor, lin):
        t = ((to_tensor >> batch >> lin) / (to_tensor >> batch >> lin)
             + (to_tensor >> batch) / (to_tensor >> batch)
        ) - 'name'
        g = t + Identity()
        t_preprocess = group(group(to_tensor / to_tensor) + group(to_tensor / to_tensor)) - 'name'
        g_preprocess = group(t_preprocess + Identity())
        t_forward = group(group(lin / lin) / Identity()) - 'name'
        g_forward = group(t_forward / Identity())
        assert str(t.pd_preprocess) == str(t_preprocess)
        assert str(g.pd_preprocess) == str(g_preprocess)
        assert str(t.pd_forward) == str(t_forward)
        assert str(g.pd_forward) == str(g_forward)


class TestFunctionTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = get_info
        request.cls.transform_3 = plus_global

    def test_pd_preprocess(self):
        assert isinstance(self.transform_1.pd_preprocess, pd.Identity)

    def test_pd_forward(self):
        assert isinstance(self.transform_1.pd_forward, pd.FunctionTransform)

    def test_pd_postprocess(self):
        assert isinstance(self.transform_1.pd_postprocess, pd.Identity)

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(5) == 6
        assert self.transform_3.infer_apply(5) == 10

    def test_eval_apply(self):
        out = list(self.transform_1.eval_apply([5, 6]))
        assert len(out) == 2
        assert out[0] == 6
        assert out[1] == 7
        out = list(self.transform_2.eval_apply([{'info': 'hello'}, {'info': 'dog'}]))
        assert len(out) == 2
        assert out[0] == 'hello'
        assert out[1] == 'dog'

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
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        t3 = pd.load(tmp_path / 'test.padl')
        assert t3.infer_apply(5) == 10


def test_name():
    assert (plus_one - 'p1')._pd_name == 'p1'
    assert plus_one._pd_name is None


class TestTransformDeviceCheck:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = Batchify() >> (plus_one >> times_two) >> times_two
        request.cls.transform_2 = Batchify() >> plus_one >> (times_two >> times_two)

    def test_device_check(self):
        self.transform_1.pd_to('gpu')
        self.transform_1.transforms[1].pd_to('cpu')

        assert self.transform_1.pd_forward_device_check()

        self.transform_2.pd_to('gpu')
        assert self.transform_2.pd_forward_device_check()


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
        self.transform_1.pd_save(tmp_path / 'test')
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
        request.cls.transform_2 = \
            Polynomial(2, 3, pd_save_options={torch.nn.Module: 'no-save'})

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

    def test_pd_save_with_options(self, tmp_path, capsys):
        self.transform_2.pd_save(tmp_path / 'test.padl')
        print(tmp_path / 'test.padl')
        assert not os.path.exists((tmp_path / 'test.padl') / '0.pt')
        pd.load(tmp_path / 'test.padl')
        out, err = capsys.readouterr()
        assert 'loading torch module from' not in out


class TestTorchModuleTransformWithJit:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        # transform wrapped around torch.jit.script
        request.cls.jit_1 = transform(torch.jit.script(PolynomialClass(2, 3)))
        request.cls.compose_1 = transform(lambda x: x + 1) >> batch >> self.jit_1

    def test_type(self):
        assert isinstance(self.jit_1, pd.Transform)
        assert isinstance(self.compose_1, pd.Transform)

    def test_output(self):
        assert self.jit_1(torch.tensor(1)) == torch.tensor(2)

    def test_infer_apply(self):
        assert self.jit_1.infer_apply(torch.tensor(1)) == torch.tensor(2)
        assert self.compose_1.infer_apply(torch.tensor(0)) == torch.tensor(2)

    def test_eval_apply(self):
        assert list(self.jit_1.eval_apply(torch.tensor([1])))[0] == torch.tensor([2])
        assert list(self.compose_1.eval_apply(torch.tensor([0])))[0] == torch.tensor([2])

    def test_device(self):
        self.jit_1.pd_to('cpu')
        device = next(self.jit_1.pd_layers[0].parameters()).device.type
        assert device == 'cpu'

    def test_pd_layers(self):
        assert len(self.jit_1.pd_layers) > 0

    def test_pd_parameters(self):
        assert len(list(self.jit_1.pd_parameters())) == 2

    def test_save_and_load(self, tmp_path):
        self.jit_1.pd_save(tmp_path / 'test.padl')
        t1 = pd.load(tmp_path / 'test.padl')
        assert t1.infer_apply(torch.tensor(1)) == torch.tensor(2)

        self.compose_1.pd_save(tmp_path / 'test.padl', True)
        compose_1 = pd.load(tmp_path / 'test.padl')
        assert compose_1.infer_apply(torch.tensor(0)) == torch.tensor(2)

    def test_methods(self):
        diff = set(dir(pd.TorchModuleTransform)) - set(dir(self.jit_1))
        assert len(diff) == 0


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

    def test_long_list(self):
        import tests.material.long_list


class TestComposeWithComments:
    def test_lambda_1(self):
        # should not fail
        t = (
            transform(lambda x: x)
        #
            >> transform(lambda x: x)
        )

    def test_lambda_2(self):
        # should not fail
        t = (
            transform(lambda x: x)
            #
            >> transform(lambda x: x)
        )

    def test_identity(self):
        # should not fail
        t = (
            Identity()
        #
            >> transform(lambda x: x)
        )

    def test_function_1(self, tmp_path):
        t = (
            Identity()
            #
            >> transform(simple_func)
        )

        t.pd_save(tmp_path)

    def test_function_2(self, tmp_path):
        t = (
            Identity()
        #
            >> transform(simple_func)
        )

        t.pd_save(tmp_path)


class TestAssertNoDoubleBatch:
    def test_double_1(self):
        with pytest.raises(AssertionError):
            t = plus_one >> batch >> batch
            t.pd_forward

    def test_double_2(self):
        with pytest.raises(AssertionError):
            t = plus_one >> plus_one + batch >> plus_one >> batch >> plus_one
            t.pd_forward

    def test_double_3(self):
        with pytest.raises(AssertionError):
            t = plus_one >> plus_one + batch >> plus_one >> batch + plus_one >> plus_one
            t.pd_forward

    def test_double_4(self):
        with pytest.raises(AssertionError):
            t = plus_one >> plus_one / batch >> batch + plus_one >> plus_one
            t.pd_forward

    def test_no_double(self):
        t = plus_one >> plus_one / batch >> batch / plus_one >> plus_one
        t.pd_forward


class TestTrace:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        emb = transform(torch.nn.Embedding)(10, 8)
        linear = transform(torch.nn.Linear)(4, 4)
        to_tensor = transform(lambda x: torch.LongTensor(x))
        request.cls.pipeline = to_tensor >> batch >> emb >> linear

    def test_pd_trace(self):
        try:
            list(self.pipeline.train_apply([[9, 8, 8], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
                                           batch_size=2, num_workers=0))
        except:
            from padl.transforms import _pd_trace
            assert len(_pd_trace) == 3
            assert _pd_trace[0].error_position == 0
            assert torch.equal(_pd_trace[1].args, torch.LongTensor([[9, 8, 8], [4, 4, 4]]))
            assert _pd_trace[1].error_position == 1
            assert _pd_trace[1].pd_mode == 'train'
            assert _pd_trace[2].args == [[9, 8, 8], [4, 4, 4]]


def test_identity_compose_saves(tmp_path):
    t = padl.identity >> padl.identity
    t.pd_save(tmp_path / 'test')


class TestParam:
    def test_param_works(self, tmp_path):
        x = padl.param(1, 'x')
        t = SimpleClassTransform(x)
        assert t(1) == 2
        t.pd_save(tmp_path / 'test.padl')
        t_1 = padl.load(tmp_path / 'test.padl')
        assert t_1(1) == 2
        t_2 = padl.load(tmp_path / 'test.padl', x=2)
        assert t_2(1) == 3

    def test_no_default(self, tmp_path):
        x = padl.param(1, 'x', use_default=False)
        t = SimpleClassTransform(x)
        assert t(1) == 2
        t.pd_save(tmp_path / 'test.padl')
        with pytest.raises(ValueError):
            padl.load(tmp_path / 'test.padl')
        t_2 = padl.load(tmp_path / 'test.padl', x=2)
        assert t_2(1) == 3

    def test_wrong_param(self, tmp_path):
        x = padl.param(1, 'x')
        t = SimpleClassTransform(x)
        assert t(1) == 2
        t.pd_save(tmp_path / 'test.padl')
        with pytest.raises(ValueError):
            padl.load(tmp_path / 'test.padl', y=1)
        t_2 = padl.load(tmp_path / 'test.padl', x=2)
        assert t_2(1) == 3


def test_device_check_in_init_works():
    from tests.material.transforms_in_module import DeviceCheckInInit
    t = SimpleClassTransform(1)
    DeviceCheckInInit(t >> t >> batch >> t)  # should not cause an error

