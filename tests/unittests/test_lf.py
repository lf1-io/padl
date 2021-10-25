import pytest
from lf import transform as lf, trans
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


@trans
def trans_with_globals(x, y):
    return (plus >> times_two)(x, y)


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

    def test_infer_apply(self):
        assert self.transform_4.infer_apply(1) == 4

    def test_all_transforms_1(self):
        c = plus_one >> times_two >> times_two
        all_ = c.lf_all_transforms()
        assert set(all_) == set([plus_one, times_two, c])

    def test_all_transforms_2(self):
        c = plus_one >> times_two >> trans_with_globals
        all_ = c.lf_all_transforms()
        assert set(all_) == set([plus_one, times_two, c, trans_with_globals, plus])


class TestFunctionTransform:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = plus_one
        request.cls.transform_2 = get_info

    def test_preprocess(self):
        assert isinstance(self.transform_1.lf_preprocess, lf.Identity)

    def test_postprocess(self):
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

    def test_all_transforms(self):
        all_ = trans_with_globals.lf_all_transforms()
        assert set(all_) == set([plus, times_two, trans_with_globals])
