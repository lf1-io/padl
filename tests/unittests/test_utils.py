from unittest import TestCase

from lf import utils
from lf import transforms as lf
from lf.testing import assert_close
import torch


class TestBatchget:
    def test_a(self):
        a = torch.Tensor([1, 2, 3])
        assert utils.batchget(a, 0) == 1

    def test_b(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        assert utils.batchget([a, b], 0) == (1, 4)

    def test_c(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        c = torch.Tensor([7, 8, 9])
        assert utils.batchget([a, b, c], 0) == (1, 4, 7)

    def test_d(self):
        a = torch.Tensor([1, 2, 3])
        b = torch.Tensor([4, 5, 6])
        c = torch.Tensor([7, 8, 9])
        assert utils.batchget([a, [b, c]], 0) == (1, (4, 7))


class TestUnbatch(TestCase):
    def test_a(self):
        a = torch.Tensor(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        r = utils.unbatch(a)
        assert_close(r[0], [1, 2, 3])
        assert_close(r[1], [4, 5, 6])

    def test_b(self):
        a = (
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            ),
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            )
        )
        r = utils.unbatch(a)
        assert_close(r[0], ([1, 2, 3], [1, 2, 3]))
        assert_close(r[1], ([4, 5, 6], [4, 5, 6]))

    def test_c(self):
        a = (
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            ), (
                torch.Tensor(
                    [[1, 2, 3],
                     [4, 5, 6]]
                ),
                torch.Tensor(
                    [[1, 2, 3],
                     [4, 5, 6]]
                )
            )
        )
        r = utils.unbatch(a)
        assert_close(r[0], ([1, 2, 3], ([1, 2, 3], [1, 2, 3])))
        assert_close(r[1], ([4, 5, 6], ([4, 5, 6], [4, 5, 6])))


class TestMakeBatch(TestCase):
    def test_a(self):
        a = torch.Tensor(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        r = utils.unbatch(a)
        b = utils.make_batch(r)
        assert_close(b, a)

    def test_b(self):
        a = (
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            ),
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            )
        )
        r = utils.unbatch(a)
        b = utils.make_batch(r)
        assert_close(b, a)

    def test_c(self):
        a = (
            torch.Tensor(
                [[1, 2, 3],
                 [4, 5, 6]]
            ), (
                torch.Tensor(
                    [[1, 2, 3],
                     [4, 5, 6]]
                ),
                torch.Tensor(
                    [[1, 2, 3],
                     [4, 5, 6]]
                )
            )
        )
        r = utils.unbatch(a)
        b = utils.make_batch(r)
        assert_close(b, a)


class TestSubsample(TestCase):
    def test_a(self):
        # same seed should always return this
        s = utils.subsample(list(range(100)), 10, seed=1)
        assert list(s) == [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]

    def test_b(self):
        s = utils.subsample(list(range(100)), 10, seed=123)
        assert list(s) != [2, 9, 6, 4, 0, 3, 1, 7, 8, 5]

    def test_c(self):
        r = list(range(100))
        s = utils.subsample(r, 1000, seed=123)
        assert r == s


class TestEval(TestCase):
    def test_a(self):
        m = lf.Transform()
        m.train()
        assert m.stage, 'train'

        with utils.eval(m):
            assert m.stage == 'eval'

        assert m.stage == 'train'


class TestStage(TestCase):
    def test_a(self):
        m = lf.Transform()
        m.train()
        assert m.stage == 'train'

        with utils.stage(m, 'eval'):
            assert m.stage == 'eval'

        assert m.stage == 'train'

    def test_b(self):
        m = lf.Transform()
        m.train()
        assert m.stage == 'train'

        with utils.stage(m, 'infer'):
            assert m.stage == 'infer'

        assert m.stage == 'train'


class TestEvalModel(TestCase):
    def test_a(self):
        m = torch.nn.Linear(1, 2)
        m.train()
        assert m.training

        with utils.evalmodel(m):
            assert not m.training

        assert m.training
