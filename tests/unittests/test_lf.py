from lf import lf


def plus_one_(x):
    return x + 1


def test_plus_one():
    plus_one = lf.trans(plus_one_)
    assert plus_one.stage is None
    assert plus_one.infer_apply(5) == 6
