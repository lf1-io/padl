import pytest
import torch
from padl import transform, Batchify
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


class TestIfInStage:
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

    def test_infer_apply_1(self):
        assert self.transform_1.infer_apply(1) == 9

    def test_infer_apply_2(self):
        assert self.transform_2.infer_apply(1) == 6

    def test_infer_apply_3(self):
        assert self.transform_3.infer_apply(1) == 12

    def test_eval_apply_1(self):
        assert list(self.transform_1.eval_apply([1, 2])) == [9, 12]

    def test_eval_apply_2(self):
        assert list(self.transform_2.eval_apply([1, 2])) == [12, 18]

    def test_eval_apply_3(self):
        assert list(self.transform_3.eval_apply([1, 2])) == [6, 9]

    def test_train_apply_1(self):
        assert list(self.transform_1.train_apply([1, 2])) == [12, 18]

    def test_train_apply_2(self):
        assert list(self.transform_2.train_apply([1, 2])) == [6, 9]

    def test_train_apply_3(self):
        assert list(self.transform_3.train_apply([1, 2])) == [6, 9]

    def test_save_and_load_1(self, tmp_path):
        self.transform_1.pd_save(tmp_path / 'test.padl')
        t1 = padl.load(tmp_path / 'test.padl')
        assert t1.infer_apply(1) == 9

    def test_save_and_load_2(self, tmp_path):
        self.transform_2.pd_save(tmp_path / 'test.padl')
        t2 = padl.load(tmp_path / 'test.padl')
        assert t2.infer_apply(1) == 6

    def test_save_and_load_3(self, tmp_path):
        self.transform_3.pd_save(tmp_path / 'test.padl')
        t3 = padl.load(tmp_path / 'test.padl')
        assert t3.infer_apply(1) == 12


class TestTry:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.to_int = transform(lambda x: 1)
        request.cls.str_sum = transform(lambda x: x + 'a')
        request.cls.return_zero = transform(lambda x: 0)
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
        request.cls.try_transform_3 = (
            plus_one
            >> transform(lambda x: torch.LongTensor([x]))
            >> Batchify()
            >> Try(TensorMult(3), TensorDivide('a'), TypeError, TensorDivide('a'), TensorMult(2))
        )

    def test_infer_apply_1(self):
        assert self.try_transform_1.infer_apply(4).item() == 10

    def test_infer_apply_2(self):
        assert self.try_transform_2.infer_apply(3).item() == 16

    def test_infer_apply_3(self):
        assert self.try_transform_3.infer_apply(2).item() == 18

    def test_train_apply_1(self):
        assert list(self.try_transform_1.train_apply([5, 8])) == [12, 18]

    def test_train_apply_2(self):
        assert list(self.try_transform_2.train_apply([2, 3, 4])) == [12, 16, 20]

    def test_train_apply_3(self):
        assert list(self.try_transform_3.train_apply([1, 5, 2])) == [12, 36, 18]

    def test_eval_apply_1(self):
        assert list(self.try_transform_1.eval_apply([4, 9])) == [10, 20]

    def test_eval_apply_2(self):
        assert list(self.try_transform_2.eval_apply([3, 7])) == [16, 32]

    def test_eval_apply_3(self):
        assert list(self.try_transform_3.eval_apply([2, 6])) == [18, 42]

    def test_save_and_load_1(self, tmp_path):
        self.try_transform_1.pd_save(tmp_path / 'test.padl')
        t1 = padl.load(tmp_path / 'test.padl')
        assert t1.infer_apply(4) == 10

    def test_save_and_load_2(self, tmp_path):
        self.try_transform_2.pd_save(tmp_path / 'test.padl')
        t2 = padl.load(tmp_path / 'test.padl')
        assert t2.infer_apply(3) == 16

    def test_save_and_load_3(self, tmp_path):
        self.try_transform_3.pd_save(tmp_path / 'test.padl')
        t3 = padl.load(tmp_path / 'test.padl')
        assert t3.infer_apply(2) == 18
