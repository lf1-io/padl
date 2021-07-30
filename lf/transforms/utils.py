# pylint: disable=no-member,arguments-differ,not-callable
"""Utilities for lf transforms"""
import functools
import torch

from lf.transforms.core import Layer
import lf.typing.types as t


class _TorchNN:
    """ Helper class for wrapping modules from torch.nn into Layer.

    >> tnn = TorchNN()
    >> tnn.Linear(10,20)
    """
    types = {
        torch.nn.Dropout: lambda l: (t.Tensor('?a'), t.Tensor('?a')),
        torch.nn.ReLU: lambda l: (t.Tensor('?a'), t.Tensor('?a')),
        torch.nn.Linear: lambda l: (t.Tensor(shape=['?b', l.in_features]),
                                    t.Tensor(shape=['?b', l.out_features]))
    }

    def __getattr__(self, attr):
        layer_class = getattr(torch.nn, attr)

        class C:
            """
            Helper class
            """
            def __init__(inner_self):

                inner_self.in_type = None
                inner_self.out_type = None

            def __getitem__(inner_self, x):
                if isinstance(x, tuple):
                    assert len(x) == 2
                    inner_self.in_type, inner_self.out_type = x
                else:
                    inner_self.in_type = x
                if isinstance(inner_self.in_type, tuple):
                    inner_self.in_type = t.Tensor(shape=list(inner_self.in_type))
                if isinstance(inner_self.out_type, tuple):
                    inner_self.out_type = t.Tensor(shape=list(inner_self.out_type))
                return inner_self

            def __call__(inner_self, *args, **kwargs):
                layer = layer_class(*args, **kwargs)
                if inner_self.in_type is None or inner_self.out_type is None:
                    in_type, out_type = self.types.get(
                        layer_class, lambda x: (t.Any(), t.Any()))(layer)
                if inner_self.in_type is not None:
                    in_type = inner_self.in_type
                if inner_self.out_type is not None:
                    out_type = inner_self.out_type
                return Layer(layer, layer_name=f'{layer}', in_type=in_type, out_type=out_type)

        return functools.wraps(layer_class)(C())


tnn = _TorchNN()
