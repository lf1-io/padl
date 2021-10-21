import torch
from lf import lf
from lf.util_transforms import Batchify


def plus_one_(x):
    return x + 1


def get_info_(x):
    return x['info']


def test_context():
    plus_one = lf.trans(plus_one_)
    assert plus_one.lf_stage is None
    with plus_one.lf_set_stage('infer'):
        assert plus_one.lf_stage is 'infer'
        assert plus_one.lf_preprocess.lf_stage == 'infer'
        assert plus_one.lf_forward.lf_stage == 'infer'
        assert plus_one.lf_postprocess.lf_stage == 'infer'


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


def test_compose():
    plus_one = lf.trans(plus_one_)
    comp1 = lf.Compose([plus_one, plus_one], module=None, stack=None)
    assert comp1(2) == 4
    assert comp1.infer_apply(2) == 4

    plus_one = lf.trans(plus_one_)
    comp2 = lf.Compose([plus_one, plus_one, Batchify()], module=None, stack=None)
    print(comp2.infer_apply(2))
    print(list(comp2.eval_apply([2, 2])))
    print(list(comp2.train_apply([2, 2])))


# TODO Add back once I can test Compose with preprocess step
# def test_loader_kwargs():
#     plus_one = lf.trans(plus_one_)
#     loader_kwargs = {'batch_size': 2}
#     out = list(plus_one.eval_apply([5, 6, 7, 8], loader_kwargs=loader_kwargs, flatten=False))
#     assert torch.all(out[0] == torch.tensor([6, 7]))
#     assert torch.all(out[1] == torch.tensor([8, 9]))
