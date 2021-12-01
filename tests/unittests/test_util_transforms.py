import pytest
import torch
from padl import transform, Batchify, batch, Identity
import padl.transforms as padl
from padl.util_transforms import IfTrain, IfEval, IfInfer, Try


@transform
def plus_one(x):
    return x + 1


@transform
def times_two(x):
    return x * 2


times_three = transform(lambda x: x * 3)


@transform
class TensorMult(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return self.factor * x


@transform
class TensorDivide(torch.nn.Module):
    def __init__(self, div):
        super().__init__()
        self.div = div

    def forward(self, x):
        return x / self.div


class TestIfInMode:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (
            plus_one
            >> IfTrain(times_two, plus_one)
            >> Batchify()
            >> times_three
        )
        request.cls.transform_2 = (
            plus_one
            >> IfEval(times_two)
            >> Batchify()
            >> times_three
        )
        request.cls.transform_3 = (
            plus_one
            >> IfInfer(times_two)
            >> Batchify()
            >> times_three
        )

    def test_infer_apply(self):
        assert self.transform_1.infer_apply(1) == 9
        assert self.transform_2.infer_apply(1) == 6
        assert self.transform_3.infer_apply(1) == 12

    def test_eval_apply(self):
        assert list(self.transform_1.eval_apply([1, 2])) == [9, 12]
        assert list(self.transform_2.eval_apply([1, 2])) == [12, 18]
        assert list(self.transform_3.eval_apply([1, 2])) == [6, 9]

    def test_train_apply(self):
        assert list(self.transform_1.train_apply([1, 2])) == [12, 18]
        assert list(self.transform_2.train_apply([1, 2])) == [6, 9]
        assert list(self.transform_3.train_apply([1, 2])) == [6, 9]

    def test_save_and_load(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl', True)
        t1 = padl.load(tmp_path / 'test.padl')
        assert t1.infer_apply(1) == 9
        self.transform_2.pd_save(tmp_path / 'test.padl', True)
        t2 = padl.load(tmp_path / 'test.padl')
        assert t2.infer_apply(1) == 6
        self.transform_3.pd_save(tmp_path / 'test.padl', True)
        t3 = padl.load(tmp_path / 'test.padl')
        assert t3.infer_apply(1) == 12


class TestTry:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        tmp = transform(lambda x: x / 0)
        request.cls.try_transform_1 = (
            Try(tmp,
                times_two,
                (TypeError, ZeroDivisionError),
                plus_one,
                plus_one)
            >> Batchify()
            >> plus_one
        )
        request.cls.try_transform_2 = (
            plus_one
            >> transform(lambda x: torch.LongTensor([x]))
            >> Batchify()
            >> Try(TensorDivide('a'), TensorMult(2), TypeError, TensorMult(2), TensorMult(2))
        )

    def test_infer_apply(self):
        assert self.try_transform_1.infer_apply(4).item() == 9
        assert self.try_transform_2.infer_apply(3).item() == 8

    def test_train_apply(self):
        assert list(self.try_transform_1.train_apply([5, 8])) == [11, 17]
        assert list(self.try_transform_2.train_apply([2, 3, 4])) == [6, 8, 10]

    def test_eval_apply(self):
        assert list(self.try_transform_1.eval_apply([4, 9])) == [9, 19]
        assert list(self.try_transform_2.eval_apply([3, 7])) == [8, 16]

    def test_stages_1(self):
        tt = self.try_transform_1[0]
        assert isinstance(tt.pd_preprocess, Identity)
        assert tt.pd_forward is tt
        assert isinstance(tt.pd_postprocess, Identity)

    def test_no_multistage_try(self):
        with pytest.raises(AssertionError):
            t = Try(plus_one >> batch >> plus_one, Identity(), exceptions=Exception)
            t.pd_forward

    def test_no_multistage_catch(self):
        with pytest.raises(AssertionError):
            t = Try(Identity(), plus_one >> batch >> plus_one,
                    exceptions=Exception)
            t.pd_forward

    def test_no_multistage_else(self):
        with pytest.raises(AssertionError):
            t = Try(Identity(), Identity(), else_transform=plus_one >> batch >> plus_one,
                    exceptions=Exception)
            t.pd_forward

    def test_save_and_load(self, tmp_path):
        self.try_transform_1.pd_save(tmp_path / 'test.padl', force_overwrite=True)
        t1 = padl.load(tmp_path / 'test.padl')
        assert t1.infer_apply(4) == 9

        self.try_transform_2.pd_save(tmp_path / 'test.padl', force_overwrite=True)
        t2 = padl.load(tmp_path / 'test.padl')
        assert t2.infer_apply(3) == 8
