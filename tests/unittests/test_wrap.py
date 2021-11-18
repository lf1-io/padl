import pytest

from padl import load, transform
from padl.transforms import ClassTransform, FunctionTransform

import numpy as np
import torch


def test_raise():
    with pytest.raises(ValueError):
        transform(2)


np = transform(np)
pd_torch = transform(torch)


def test_function_a(tmp_path):
    t = np.cos
    assert isinstance(t, FunctionTransform)
    assert t._pd_call == 'np.cos'
    t.pd_save(tmp_path / 'test.padl', True)
    t_ = load(tmp_path / 'test.padl')
    assert t_.infer_apply(1.3) == t.infer_apply(1.3)


def test_function_b():
    t = np.random.rand
    assert isinstance(t, FunctionTransform)
    assert t._pd_call == 'np.random.rand'


def test_class_a():
    t = pd_torch.nn.Linear(10, 10)
    assert isinstance(t, ClassTransform)
    assert t._pd_call == 'pd_torch.nn.Linear(10, 10)'


class TestModuleWrap:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = np.sin
        request.cls.transform_2 = (np.sin >> np.sin)
        transform_temp = np.sin
        request.cls.transform_3 = transform_temp + transform_temp >> np.add
        request.cls.transform_4 = np.cos + np.cos >> np.add

    def test_save_load(self, tmp_path):
        for transform_ in [self.transform_1, self.transform_2, self.transform_3, self.transform_4]:
            transform_.pd_save(tmp_path / 'test.padl', True)
            t_ = load(tmp_path / 'test.padl')
            assert t_.infer_apply(1.3) == transform_.infer_apply(1.3)
