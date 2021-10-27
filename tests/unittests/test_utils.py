import torch
from lf.utils import this
<<<<<<< HEAD


def test_this():
    t = this.tolist()
    assert t(torch.tensor([1, 2, 3])) == [1, 2, 3]

    t = this[0]
    assert t([1, 2, 3]) == 1

    t = this[:2]
    assert t([1, 2, 3]) == [1, 2]
=======
from tests.fixtures.transforms import cleanup_checkpoint


def test_this(cleanup_checkpoint):
    t = this.tolist()
    t.lf_save('test.lf')
    assert t(torch.tensor([1, 2, 3])) == [1, 2, 3]

    t = this[0]
    t.lf_save('test.lf')
    assert t([1, 2, 3]) == 1

    t = this[:2]
    t.lf_save('test.lf')
    assert t([1, 2, 3]) == [1, 2]

    t = this[0]
    t.lf_save('test.lf')
    assert (t >> t).infer_apply([[0, 1], 2]) == 0
>>>>>>> 5ca1d7bea417c148064ec1ffa8e30a6f764420d7
