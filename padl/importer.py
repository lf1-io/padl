"""Special module that allows importing any other module such that all callables inside are wrapped
as transforms.

Example:

>>> from importer.torch import nn
>>> x = nn.Linear(10, 10)
>>> isinstance(x, padl.transforms.Transform)
True
"""

import importlib
import inspect
import sys
from types import MethodWrapperType, ModuleType

from padl.wrap import _wrap_class, _wrap_function


class PatchedModule:
    """Class that patches a module, such that all functions and classes in that module come out
    wrapped as Transforms.

    Example:

        >>> from padl.importer import numpy as np
        >>> isinstance(np.random.rand, Transform)
        True
    """

    def __init__(self, module, parents=None):
        self._module = module
        if parents is None:
            self._path = self._module.__name__
        else:
            self._path = parents + '.' + self._module.__name__

    def __getattr__(self, key):
        x = getattr(self._module, key)
        if inspect.isclass(x):
            if hasattr(x, '__call__') and not isinstance(x.__call__, MethodWrapperType):
                return _wrap_class(x)
            return x
        if callable(x):
            return _wrap_function(x, ignore_scope=True)
        if isinstance(x, ModuleType):
            return PatchedModule(x, parents=self._path)
        return x

    def __repr__(self):
        return f'Transform patched: {self._module}'

    def __dir__(self):
        return dir(self._module)


class _PatchFactory:
    """Class that allows patching imported modules. """

    def __getattr__(self, name):
        try:
            module = importlib.import_module(name)
            return PatchedModule(module)
        except ModuleNotFoundError:
            pass


sys.modules[__name__] = _PatchFactory()
