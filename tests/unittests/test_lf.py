import torch
from lf import lf


def plus_one_(x):
    return x + 1


def get_info_(x):
    return x['info']


def test_context():
    plus_one = lf.trans(plus_one_)
    assert plus_one.stage is None
    with plus_one.set_stage('infer'):
        assert plus_one.stage is 'infer'
        assert plus_one.preprocess.stage == 'infer'
        assert plus_one.forward.stage == 'infer'
        assert plus_one.postprocess.stage == 'infer'


def test_infer_apply():
    plus_one = lf.trans(plus_one_)
    assert plus_one.infer_apply(5) == 6


def test_eval_apply():
    plus_one = lf.trans(plus_one_)
    out = list(plus_one.eval_apply([5, 6], flatten=False))
    assert len(out) == 2
    assert out[0] == 6
    assert out[1] == 7

    get_info = lf.trans(get_info_)
    out = list(get_info.eval_apply([{'info': 'hello'}, {'info': 'dog'}], flatten=False))
    assert len(out) == 2
    assert out[0] == 'hello'
    assert out[1] == 'dog'


# TODO Add back once I can test Compose with preprocess step
# def test_loader_kwargs():
#     plus_one = lf.trans(plus_one_)
#     loader_kwargs = {'batch_size': 2}
#     out = list(plus_one.eval_apply([5, 6, 7, 8], loader_kwargs=loader_kwargs, flatten=False))
#     assert torch.all(out[0] == torch.tensor([6, 7]))
#     assert torch.all(out[1] == torch.tensor([8, 9]))
