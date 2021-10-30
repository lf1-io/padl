import pytest
from padl import transform, Batchify
import padl.transforms as padl
from padl.util_transforms import IfTrain, IfEval, IfInfer
from tests.fixtures.transforms import cleanup_checkpoint


@transform
def plus_one(x):
    return x + 1


@transform
def times_two(x):
    return x * 2


times_three = transform(lambda x: x * 3)


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

    def test_save_and_load(self, cleanup_checkpoint):
        self.transform_1.pd_save('test.padl')
        t1 = padl.load('test.padl')
        assert t1.infer_apply(1) == 9
        self.transform_2.pd_save('test.padl')
        t2 = padl.load('test.padl')
        assert t2.infer_apply(1) == 6
        self.transform_3.pd_save('test.padl')
        t3 = padl.load('test.padl')
        assert t3.infer_apply(1) == 12
