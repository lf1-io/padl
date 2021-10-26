import pytest
import torch
from lf import transform as lf, trans, Identity
from lf.transform import Batchify, Unbatchify
from lf.util_transforms import IfTrain, IfEval, IfInfer


@trans
def plus_one(x):
    return x + 1


@trans
def times_two(x):
    return x * 2


class TestIfInStage:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = (
            plus_one
            >> IfTrain(times_two, plus_one)
            >> Batchify()
            >> trans(lambda x: x * 3)
        )
        request.cls.transform_2 = (
            plus_one
            >> IfEval(times_two)
            >> Batchify()
            >> trans(lambda x: x * 3)
        )
        request.cls.transform_3 = (
            plus_one
            >> IfInfer(times_two)
            >> Batchify()
            >> trans(lambda x: x * 3)
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
