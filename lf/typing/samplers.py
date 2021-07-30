# pylint: disable=no-member
"""Typing samplers"""
import random
import numpy as np
from PIL import Image
import torch

from .patterns import free, convert, val, copy


class TensorSampler:
    """
    Sample random tensor
    :param typevar: type of var
    :return:
    """
    def __call__(self, typevar):
        shape = copy(typevar.kwargs['shape'], {})
        if free(shape.tail):
            shape.tail @ convert([random.randrange(1, 10)])
        for x in shape.iter_no_tail():
            if free(x):
                x @ random.randrange(1, 10)
        return self.random([val(x) for x in shape], typevar.kwargs)

    def random(self, shape, kwargs):
        return NotImplemented


class FloatTensorSampler(TensorSampler):
    """"FloatTensorSampler"""
    @staticmethod
    def random(shape, kwargs=None):
        return torch.randn(shape)


class LongTensorSampler(TensorSampler):
    """"LongTensorSampler"""
    @staticmethod
    def random(shape, kwargs=None):
        return torch.randint(0, 100, shape)


class NumpyFloatTensorSampler(TensorSampler):
    """NumpyFloatTensorSampler"""
    @staticmethod
    def random(shape, kwargs=None):
        return np.random.randn(*shape)


class ImageSampler(TensorSampler):
    @staticmethod
    def random(shape, kwargs=None):
        mode = copy(kwargs['mode'], {})
        if free(mode):
            mode = 'RGB'
        else:
            mode = val(mode)

        modes_dims = (
            (('1', 'L', 'P', 'I', 'F'), 1),
            (('RGB', 'YCbCr', 'LAB', 'HSV'), 3),
            (('RGBA', 'CMYK'), 4)
        )
        modes_dims = {k: v for ks, v in modes_dims for k in ks}
        modes_types = (
            (('1'),
             lambda s: np.random.randint(0, 2, s, dtype=np.bool)),
            (('L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV'),
             lambda s: np.random.randint(0, 256, s, dtype=np.uint8)),
            (('I',),
             lambda s: np.random.randint(-2**31, 2**31, s, dtype=np.int32)),
            (('F',),
             lambda s: np.random.random(shape).astype(np.float32)),
        )
        modes_types = {k: v for ks, v in modes_types for k in ks}

        if free(mode):
            mode = random.choice(modes_dims)
        else:
            mode = val(mode)

        if modes_dims[mode] > 1:
            shape = tuple(shape) + (modes_dims[mode],)
        return Image.fromarray(modes_types[mode](shape), mode=mode)
