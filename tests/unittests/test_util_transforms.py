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
        request.cls.transform_1 = plus_one >> IfTrain(times_two, plus_one)
        request.cls.transform_2 = plus_one >> IfTrain(times_two)
