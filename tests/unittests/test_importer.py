import pytest

from padl import load, transform
from padl.transforms import ClassTransform, FunctionTransform

import numpy as np
import torch

pdnp = transform(np)
pdtorch = transform(torch)


def test_function_a():
    t = pdnp.cos
    assert isinstance(t, FunctionTransform)
    assert t._pd_call == 'np.cos'


def test_function_b():
    t = pdnp.random.rand
    assert isinstance(t, FunctionTransform)
    assert t._pd_call == 'np.random.rand'


def test_class_a():
    t = pdtorch.nn.Linear(10, 10)
    assert isinstance(t, ClassTransform)
    assert t._pd_call == 'torch.nn.Linear(10, 10)'


class TestPADLImporter:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = pdnp.sin
        request.cls.transform_2 = (pdnp.sin >> pdnp.sin)
        transform_temp = pdnp.sin
        request.cls.transform_3 = transform_temp + transform_temp >> pdnp.add
        request.cls.transform_4 = pdnp.cos + pdnp.cos >> pdnp.add

    def test_save_load(self, tmp_path):
        for transform_ in [self.transform_1, self.transform_2, self.transform_3, self.transform_4]:
            transform_.pd_save(tmp_path / 'test.padl', True)
            t_ = load(tmp_path / 'test.padl')
            assert t_.infer_apply(1.3) == transform_.infer_apply(1.3)
