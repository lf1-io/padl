import importlib
import inspect
import sys
from types import ModuleType

from lf import trans
from lf.wrap import _wrap_class, _wrap_function


class PatchedModule:
    def __init__(self, module, parents=None):
        self.module = module
        if parents is None:
            self.path = self.module.__name__
        else:
            self.path = parents + '.' + self.module.__name__

    def __getattr__(self, key):
        x = getattr(self.module, key)
        if inspect.isclass(x):
            return _wrap_class(x)
        if callable(x):
            return _wrap_function(x, ignore_scope=True)
        if isinstance(x, ModuleType):
            return PatchedModule(x, parents=self.path)
        return x

    def __repr__(self):
        return f'Transform patched: {self.module}'


class Patcher:
    def __getattr__(self, name):
        try:
            m = importlib.import_module(name)
            return PatchedModule(m)
        except ModuleNotFoundError:
            pass


sys.modules[__name__] = Patcher()
