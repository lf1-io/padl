import importlib
import sys
from types import ModuleType

from lf import trans


class PatchedModule:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, key):
        x = self.module.__dict__[key]
        if callable(x):
            return trans(x)
        if isinstance(x, ModuleType):
            return PatchedModule(x)
        return x


class Patcher:
    def __getattr__(self, name):
        try:
            m = importlib.import_module(name)
            return PatchedModule(m)
        except:
            pass


sys.modules[__name__] = Patcher()
