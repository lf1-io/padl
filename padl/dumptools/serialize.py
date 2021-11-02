import ast
import json
from pathlib import Path
import sys

from padl.dumptools import inspector, sourceget, var2mod, symfinder


SCOPE = symfinder.Scope.toplevel(sys.modules[__name__])


class Serializer:
    """Serializer base class. """

    store = []
    i = 0

    def __init__(self):
        self.index = Serializer.i
        Serializer.i += 1
        self.store.append(self)

    @property
    def varname(self):
        """The varname to store in the dumped code. """
        return f'PADL_VALUE_{self.index}'

    @classmethod
    def save_all(cls, codegraph, scopemap, path):
        """Save all values. """
        for codenode in list(codegraph.values()):
            for serializer in cls.store:
                if serializer.varname in codenode.source:
                    codegraph.update(serializer.save(path))
                    for varname, scope in codenode.globals_:
                        if varname == serializer.varname:
                            scopemap[varname, scope] = SCOPE


class JSONSerializer(Serializer):
    """JSON Serializer (stores stuff as json).

    :param val: The value to serialize.
    """

    def __init__(self, val):
        self.val = val
        super().__init__()

    def save(self, path: Path):
        """Method for saving *self.val*. """
        if path is None:
            savepath = '?.json'
        else:
            savepath = path / f'{self.i}.json'
            with open(savepath, 'w') as f:
                json.dump(self.val, f)
        load_source = (
            f'with open(pathlib.Path(__file__).parent / \'{savepath.name}\') as f:\n'
            f'    {self.varname} = json.load(f)\n'
        )
        return {(self.varname, SCOPE): var2mod.CodeNode(source=load_source, globals_=set()),
                ('json', SCOPE): var2mod.CodeNode(source='import json', globals_=set(),
                                                  ast_node=ast.parse('import json').body[0]),
                ('pathlib', SCOPE): var2mod.CodeNode(source='import pathlib', globals_=set(),
                                                     ast_node=ast.parse('import pathlib').body[0])}


def _serialize(val):
    if hasattr(val, '__len__') and len(val) > 10:
        serializer = JSONSerializer(val)
        print('using json')
        return serializer.varname
    return repr(val)


def value(val):
    """Helper function that marks things in the code that should be stored by value. """
    caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
    _call, locs = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call', True)
    source = sourceget.get_source(caller_frameinfo.filename)
    sourceget.put_into_cache(caller_frameinfo.filename, sourceget.original(source),
                             _serialize(val), *locs)
    return val
