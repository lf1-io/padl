import torch
from padl.utils import this


def test_this():
    t = this.tolist()
    assert t(torch.tensor([1, 2, 3])) == [1, 2, 3]

    t = this[0]
    assert t([1, 2, 3]) == 1

    t = this[:2]
    assert t([1, 2, 3]) == [1, 2]
