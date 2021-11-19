import torch
from padl.utils import same


def test_same():
    t = same.tolist()
    assert t(torch.tensor([1, 2, 3])) == [1, 2, 3]

    t = same[0]
    assert t([1, 2, 3]) == 1

    t = same[:2]
    assert t([1, 2, 3]) == [1, 2]
