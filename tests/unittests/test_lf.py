from lf import lf


def plus_one_(x):
    return x + 1


def test_infer_apply():
    plus_one = lf.trans(plus_one_)
    assert plus_one.stage is None
    assert plus_one.infer_apply(5) == 6


def test_context():
    plus_one = lf.trans(plus_one_)
    with plus_one.set_stage('infer'):
        assert plus_one.preprocess.stage == 'infer'
        assert plus_one.forward.stage == 'infer'
        assert plus_one.postprocess.stage == 'infer'
