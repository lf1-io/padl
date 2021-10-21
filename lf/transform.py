"""The Transform class and some of its children. """
import ast
from pathlib import Path
import types
from typing import List

import torch

from lf.dumptools import var2mod, thingfinder, inspector


class _Notset:
    pass


_notset = _Notset()


class Transform:
    """Transform base class. """

    def __init__(self, call_info):
        self._lf_call_info = call_info
        self._lf_varname = _notset

    def __rshift__(self, other):
        return Compose([self, other], inspector.caller_info(), flatten=True)

    def __add__(self, other: "Transform") -> "Rollout":
        """ Rollout with *other*. """
        return Rollout([self, other], inspector.caller_info(), flatten=True)

    def __truediv__(self, other: "Transform") -> "Parallel":
        """ Parallel with *other*. """
        return Parallel([self, other], inspector.caller_info(), flatten=True)

    def lf_pre_save(self, path, i):
        """Method that is called on each transform before saving.

        :param path: The save path.
        :param i: Subtransform index.
        """

    def lf_post_load(self, path, i):
        """Method that is called on each transform after loading.

        :param path: The save path.
        :param i: Subtransform index.
        """

    def lf_save(self, path):
        """Save the transform to *path*. """
        path = Path(path)
        path.mkdir(exist_ok=True)
        for i, subtrans in enumerate(self.lf_all_transforms_with_globals()):
            subtrans.lf_pre_save(path, i)
        with open(path / 'transform.py', 'w') as f:
            f.write(self.lf_dumps())

    def _lf_codegraph_startnode(self, name):
        """Build the start-code-node - the node with the source needed to create *self* as "name".
        (in the scope where *self* was originally created).
        """
        start_source = f'{name or "_lf_dummy"} = {self.lf_evaluable_repr()}'
        start_node = ast.parse(start_source)
        start_globals = {
            (var, self._lf_call_info.scope)  # this should be the current scope ...?
            for var in var2mod.find_globals(start_node)
        }
        return var2mod.CodeNode(
            source=start_source,
            ast_node=start_node,
            globals_=start_globals
        )

    def _lf_build_codegraph(self, graph=None, scopemap=None, name=None, scope=None):
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        try:
            if self._lf_call == name:
                name = None
        except AttributeError:
            pass

        # build the start node ->
        start = self._lf_codegraph_startnode(name)
        # <-

        globals_dict = self._lf_call_info.globals
        all_vars_dict = {**globals_dict, **self._lf_call_info.nonlocals}

        # find dependencies
        todo = {*start.globals_}
        while todo and (next_ := todo.pop()):
            # we know this already - go on
            if next_ in scopemap:
                continue

            next_var, next_scope = next_

            # see if the object itself knows how to generate its codegraph
            try:
                if len(next_scope) > 0:
                    next_obj = all_vars_dict[next_var]
                else:
                    next_obj = globals_dict[next_var]
                next_obj._lf_build_codegraph(graph, scopemap, next_var)
            except (KeyError, AttributeError):
                pass
            else:
                print(next_var, 'can deal with itself')
                continue

            # find how next_var came into being
            (source, node), scope = thingfinder.find_in_scope(next_var, next_scope)
            scopemap[next_var, next_scope] = scope

            # find dependencies
            globals_ = {
                (var, scope)
                for var in var2mod.find_globals(node)
            }
            graph[next_var, scope] = var2mod.CodeNode(source=source, globals_=globals_,
                                                      ast_node=node)
            todo.update(globals_)
        # find dependencies done

        if name is not None:
            assert scope is not None
            graph[name, scope] = start
            scopemap[name, scope] = scope

        return graph, scopemap

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        """Return a string that if evaluated *in the same scope where the transform was created*
        creates the transform. """
        return NotImplemented

    def lf_all_transforms(self):
        return NotImplemented

    def lf_all_transforms_with_globals(self):
        # TODO: make work with scopes
        res = self.lf_all_transforms()
        graph, _ = self._lf_build_codegraph()
        all_globals = set()
        for v in graph.values():
            all_globals.update(v.globals_)
        for g in all_globals:
            gobj = self._lf_call_info.module.__dict__[g]
            if gobj is self:
                continue
            self._lf_all_transforms_with_globals(gobj, res)
        return res

    @staticmethod
    def _lf_all_transforms_with_globals(x, res):
        if isinstance(x, str):
            return
        if isinstance(x, Transform):
            for child_transform in x.lf_all_transforms():
                if child_transform not in res:
                    res.append(child_transform)
        try:
            for t in x:
                Transform._lf_all_transforms_with_globals(t, res)
        except TypeError:
            pass

    def lf_dumps(self):
        """Dump the transform as python code. """
        scope = thingfinder.Scope.toplevel(inspector.caller_module())
        graph, scopemap = self._lf_build_codegraph(name='_lf_main', scope=scope)
        unscoped = var2mod.unscope_graph(graph, scopemap)
        return var2mod.dumps_graph(unscoped)

    def lf_repr(self, indent=0):
        varname = self.lf_varname()
        evaluable_repr = self.lf_evaluable_repr()
        if varname is None or varname == evaluable_repr:
            return f'{evaluable_repr}'
        return f'{evaluable_repr} [{varname}]'

    def __repr__(self):
        return self.lf_repr()

    def _lf_find_varname(self, scopedict):
        try:
            return [
                k for k, v in scopedict.items()
                if v is self and not k.startswith('_')
            ][0]
        except IndexError:
            return None

    def lf_varname(self, scopedict=None):
        """The transform's variable name. """
        if self._lf_varname is _notset:
            if scopedict is None:
                scopedict = self._lf_call_info.module.__dict__
            self._lf_varname = self._lf_find_varname(scopedict)
        return self._lf_varname

    def lf_set_varname(self, val):
        self._lf_varname = val


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms, in contrast to `MetaTransform`s. """

    def __init__(self, call, call_info):
        super().__init__(call_info)
        self._lf_call = call

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        if var_transforms is None:
            var_transforms = {}
        return var_transforms.get(self, self._lf_call)

    def lf_all_transforms(self):
        res = [self]
        for v in self.__dict__.values():
            if isinstance(v, Transform):
                for child_transform in v.all_transforms():
                    if child_transform not in res:
                        res.append(child_transform)
        return res


class FunctionTransform(AtomicTransform):
    def __init__(self, function, call_info, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call, call_info)
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class TorchModuleTransform(torch.nn.Module, AtomicTransform):
    def lf_pre_save(self, path, i):
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def lf_post_load(self, path, i):
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))


class MetaTransform(Transform):
    """Abstract base class for meta-trasforms (transforms combining other transforms. """
    op = NotImplemented

    def __init__(self, transforms, call_info, flatten=True):
        super().__init__(call_info)
        if flatten:
            transforms = self._flatten_list(transforms)
        self.transforms = transforms

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        sub_reprs = [
            x.lf_varname() or x.lf_evaluable_repr(indent + 4, var_transforms)
            for x in self.transforms
        ]
        return (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )

    def _lf_build_codegraph(self, graph=None, scopemap=None, name=None, scope=None):
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        start = self._lf_codegraph_startnode(name)

        if name is not None:
            assert scope is not None
            graph[name, scope] = start
            scopemap[name, scope] = scope

        for transform in self.transforms:
            varname = transform.lf_varname()
            transform._lf_build_codegraph(graph, scopemap, varname,
                                          self._lf_call_info.scope)

        return graph, scopemap

    def lf_repr(self, indent=0):
        sub_reprs = [
            x.lf_repr(indent + 4)
            for x in self.transforms
        ]
        res = (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )
        if self.lf_varname() is not None and self.lf_varname() is not _notset:
            res += f' [{self.lf_varname()}]'
        return res

    @classmethod
    def _flatten_list(cls, transform_list: List[Transform]):
        """Flatten *list_* such that members of *cls* are not nested.

        :param list_: List of transforms.
        """
        list_flat = []

        for transform in transform_list:
            if isinstance(transform, cls):
                if transform.lf_varname is None:
                    list_flat += transform.transforms
                else:
                    list_flat.append(transform)
            else:
                list_flat.append(transform)

        return list_flat

    def lf_all_transforms(self):
        res = [self]
        for transform in self.transforms:
            for child_transform in transform.lf_all_transforms():
                if child_transform not in res:
                    res.append(child_transform)
        return res


class Compose(MetaTransform):
    op = '>>'


class Rollout(MetaTransform):
    op = '+'


class Parallel(MetaTransform):
    op = '/'


def save(transform: Transform, path):
    transform.lf_save(path)


def load(path):
    """Load transform (as saved with lf.save) from *path*. """
    path = Path(path)
    with open(path / 'transform.py') as f:
        source = f.read()
    module = types.ModuleType('lfload')
    module.__dict__.update({
        '_lf_source': source,
        '_lf_module': module,
        '__file__': str(path / 'transform.py')
    })
    exec(source, module.__dict__)
    transform = module._lf_main
    for i, subtrans in enumerate(transform.lf_all_transforms_with_globals()):
        subtrans.lf_post_load(path, i)
    return transform
