from lf.transform import ClassTransform, FunctionTransform

from lf.importer import numpy as np
from lf.importer import torch


def test_function_a():
    t = np.cos
    assert isinstance(t, FunctionTransform)
    assert t._lf_call == 'np.cos'


def test_function_b():
    t = np.random.rand
    assert isinstance(t, FunctionTransform)
    assert t._lf_call == 'np.random.rand'


def test_class_a():
    t = torch.nn.Linear(10, 10)
    assert isinstance(t, ClassTransform)
    assert t._lf_call == 'torch.nn.Linear(10, 10)'
