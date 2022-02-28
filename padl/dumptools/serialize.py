import ast
from collections.abc import Iterable
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional

from padl.dumptools import inspector, sourceget, var2mod, symfinder
from padl.dumptools.symfinder import ScopedName
from padl.dumptools.var2mod import CodeNode, CodeGraph


SCOPE = symfinder.Scope.toplevel(sys.modules[__name__])


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
        self.scope = symfinder.Scope.toplevel(module)
        self.load_codegraph = \
            var2mod.CodeGraph.build(ScopedName(load_function.__name__,
                                               symfinder.Scope.toplevel(load_function.__module__)))
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
             ScopedName(self.varname, self.scope):
                 CodeNode(source=f'{self.varname} = {self.load_name}({complete_path})',
                          globals_={ScopedName(self.load_name, self.scope)},
                          scope=self.scope,
                          name=self.varname),
             ScopedName('pathlib', SCOPE):
                 CodeNode(source='import pathlib',
                          globals_=set(),
                          ast_node=ast.parse('import pathlib').body[0],
                          scope=self.scope,
                          name='pathlib')}
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


def param(val, name, description=None, use_default=True):
    """Helper function for marking parameters.

    Parameters can be overridden when loading. See also :func:`padl.load`.

    :param val: The default value of the parameter / the value before saving.
    :param name: The name of the parameter.
    :param use_default: If True, will use *val* when loading without specifying a different value.
    :returns: *val*
    """
    caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
    if not use_default and val is not None:
        call, locs = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call', True)
        source = sourceget.get_source(caller_frameinfo.filename)
        call, args = symfinder.split_call(call)
        args = 'None, ' + args.split(',', 1)[1]
        sourceget.put_into_cache(caller_frameinfo.filename, sourceget.original(source),
                                 f'{call}({args})', *locs)

    module = inspector._module(caller_frameinfo.frame)
    if not getattr(module, '_pd_is_padl_file', False):
        return val

    module._pd_found_params[name] = val

    try:
        return module._pd_params[name]
    except KeyError as exc:
        if val is None and not use_default:
            raise ValueError(f'Unfilled parameter *{name}*. \n\n'
                             'When loading a transform, '
                             f'provide *{name}* as a keyword '
                             f'argument: padl.load(..., {name}=...).'
                             + (description is not None
                             and f'\n\nDescription: "{description}"'
                             or '')
                             ) from exc
        return val
