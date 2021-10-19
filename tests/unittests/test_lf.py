import torch
from lf import lf


def plus_one_(x):
    return x + 1


def get_info_(x):
    return x['info']


def test_context():
    plus_one = lf.trans(plus_one_)
    with plus_one.set_stage('infer'):
        assert plus_one.preprocess.stage == 'infer'
        assert plus_one.forward.stage == 'infer'
        assert plus_one.postprocess.stage == 'infer'


def test_infer_apply():
    plus_one = lf.trans(plus_one_)
    assert plus_one.stage is None
    assert plus_one.infer_apply(5) == 6


def test_eval_apply():
    plus_one = lf.trans(plus_one_)
    assert plus_one.stage is None
    out = list(plus_one.eval_apply([5, 6], flatten=False))
    assert len(out) == 2
    assert out[0] == torch.tensor([6])
    assert out[1] == torch.tensor([7])

    get_info = lf.trans(get_info_)
    out = list(get_info.eval_apply([{'info': 'hello'}, {'info': 'dog'}], flatten=False))
    assert len(out) == 2
    assert out[0] == ['hello']
    assert out[1] == ['dog']
