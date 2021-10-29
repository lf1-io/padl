from tadl.transforms import ClassTransform, FunctionTransform

from tadl.importer import numpy as np
from tadl.importer import torch


def test_function_a():
    t = np.cos
    assert isinstance(t, FunctionTransform)
    assert t._td_call == 'np.cos'


def test_function_b():
    t = np.random.rand
    assert isinstance(t, FunctionTransform)
    assert t._td_call == 'np.random.rand'


def test_class_a():
    t = torch.nn.Linear(10, 10)
    assert isinstance(t, ClassTransform)
    assert t._td_call == 'torch.nn.Linear(10, 10)'
