import ast
from collections.abc import Iterable
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional

from padl.dumptools import inspector, sourceget, var2mod
from padl.dumptools.symfinder import Scope, ScopedName
from padl.dumptools.var2mod import CodeNode, CodeGraph


SCOPE = Scope.toplevel(sys.modules[__name__])


class Serializer:
    """Serializer base class.

    :param val: The value to serialize.
    :param save_function: The function to use for saving *val*.
    :param load_function: The function to use for loading *val*.
    :param file_suffix: If set, a string that will be appended to the path.
    :param module: The module the serializer functions are defined in. Optional, default is to
        use the calling module.
    """

    store: List = []
    i: int = 0

    def __init__(self, val: Any, save_function: Callable, load_function: callable,
                 file_suffix: Optional[str] = None, module: Optional[ModuleType] = None):
        self.index = Serializer.i
        Serializer.i += 1
        self.store.append(self)
        self.val = val
        self.save_function = save_function
        self.file_suffix = file_suffix
        if module is None:
            module = inspector.caller_module()
        self.scope = Scope.toplevel(module)
        self.load_codegraph = \
            var2mod.CodeGraph().build(ScopedName(load_function.__name__,
                                                 Scope.toplevel(load_function.__module__)))
        self.load_name = load_function.__name__
        super().__init__()

    def save(self, path: Path):
        """Save the serializer's value to *path*.

        Returns a codegraph containing code needed to load the value.
        """
        if path is None:
            path = Path('?')
        if self.file_suffix is not None:
            path = Path(str(path) + f'/{self.i}{self.file_suffix}')
        filename = self.save_function(self.val, path)
        if filename is None:
            assert self.file_suffix is not None, ('if no file file_suffix is passed to *value*, '
                                                  'the *save*-function must return a filename')
            filename = path.name
        if isinstance(filename, (str, Path)):
            complete_path = f"pathlib.Path(__file__).parent / '{filename}'"
        elif isinstance(filename, Iterable):
            complete_path = ('[pathlib.Path(__file__).parent / filename for filename in ['
                             + ', '.join(f"'{fn}'"
                                         for fn in filename)
                             + ']]')
        else:
            raise ValueError('The save function must return a filename, a list of filenames or '
                             'nothing.')
        return CodeGraph(
            {**self.load_codegraph,
             ScopedName(self.varname, Scope.empty(), pos='injected'):
                 CodeNode(source=f'{self.varname} = {self.load_name}({complete_path})',
                          globals_={ScopedName(self.load_name, self.scope)},
                          ast_node=ast.parse(f'{self.varname} = {self.load_name}({complete_path})').body[0],
                          name=ScopedName(self.varname, self.scope, pos='injected')),
             ScopedName('pathlib', SCOPE, pos='injected'):
                 CodeNode(source='import pathlib',
                          globals_=set(),
                          ast_node=ast.parse('import pathlib').body[0],
                          name=ScopedName('pathlib', SCOPE, pos='injected'))}
        )

    @property
    def varname(self):
        """The varname to store in the dumped code. """
        return f'PADL_VALUE_{self.index}'

    @classmethod
    def save_all(cls, codegraph, path):
        """Save all values. """
        for codenode in list(codegraph.values()):
            for serializer in cls.store:
                if serializer.varname in codenode.source:
                    loader_graph = serializer.save(path)
                    codegraph.update(loader_graph)


def save_json(val, path):
    """Saver for json. """
    with open(path, 'w') as f:
        json.dump(val, f)


def load_json(path):
    """Loader for json. """
    with open(path) as f:
        return json.load(f)


def json_serializer(val):
    """Create a json serializer for *val*. """
    return Serializer(val, save_json, load_json, '.json', sys.modules[__name__])


def _serialize(val, serializer=None):
    if serializer is not None:
        return Serializer(val, *serializer).varname
    if hasattr(val, '__len__') and len(val) > 10:
        return json_serializer(val).varname
    return repr(val)


def value(val, serializer=None):
    """Helper function that marks things in the code that should be stored by value. """
    caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
    _call, locs = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call', True)
    source = sourceget.get_source(caller_frameinfo.filename)
    sourceget.put_into_cache(caller_frameinfo.filename, sourceget.original(source),
                             _serialize(val, serializer=serializer), *locs)
    return val
