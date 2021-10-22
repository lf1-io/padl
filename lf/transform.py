"""The Transform class and some of its children. """
import ast
from collections import Counter, namedtuple
import contextlib
import inspect
from pathlib import Path
import types
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lf.data import SimpleIterator
from lf.dumptools import var2mod, thingfinder, inspector


class _Notset:
    pass


_notset = _Notset()


def _isinstance_of_namedtuple(arg):
    """Check if input is instance of namedtuple"""
    typ = type(arg)
    base = typ.__bases__
    if len(base) != 1 or base[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(field, str) for field in fields)


class Transform:
    """Transform base class.

    :param call_info:
    :param name: name of the transform
    """
    _lf_stage = None

    def __init__(self, call_info, name=None):
        self._lf_call_info = call_info
        self._lf_varname = _notset
        self._lf_name = name
        self._lf_mapdevice = {'gpu'}
        self._lf_device = 'gpu'
        self._lf_layers = None

    @property
    def lf_name(self):
        if self._lf_name is None:
            return self.lf_varname()
        return self._lf_name

    def __rshift__(self, other):
        return Compose([self, other], inspector.caller_info())

    def __add__(self, other: "Transform") -> "Rollout":
        """Rollout with *other*. """
        return Rollout([self, other], inspector.caller_info())

    def __truediv__(self, other: "Transform") -> "Parallel":
        """Parallel with *other*. """
        return Parallel([self, other], inspector.caller_info())

    def __sub__(self, transform_name: str) -> "Transform":
        """Name the Transform"""
        return self.lf_clone(lf_name=transform_name)

    def lf_clone(self, **kwargs):
        """Clone Transform"""
        return NotImplemented

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

    def __call__(self, arg):
        return NotImplementedError

    def _lf_call_transform(self, arg):
        """Call transform with possibility to pass multiple arguments"""
        signature_parameters = inspect.signature(self).parameters
        if len(signature_parameters) == 1:
            return self(arg)
        return self(*arg)

    def _lf_callyield(self, args, stage, loader_kwargs=None, verbose=False, flatten=False):
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param loader_kwargs: data loader keyword arguments
        :param verbose: if *True* print progress bar
        :param flatten: flatten the output
        """
        assert stage in ('eval', 'train'), '_lf_callyield can only be used with stage eval or train'

        if stage == 'eval':
            torch_context = torch.no_grad()
        else:
            torch_context = contextlib.suppress()

        with self.lf_set_stage(stage), torch_context:
            preprocess = self.lf_preprocess
            forward = self.lf_forward
            post = self.lf_postprocess

            use_preprocess = not preprocess.lf_is_identity
            use_post = not post.lf_is_identity
            use_forward = not forward.lf_is_identity

            if use_preprocess:
                iterator = SimpleIterator(
                    args,
                    self.lf_preprocess.lf_to('cpu')._lf_call_transform
                )
                if loader_kwargs is None:
                    loader_kwargs = {}
                loader = self._lf_get_loader(iterator=iterator, loader_kwargs=loader_kwargs)
            else:
                loader = args

            pbar = None
            if verbose:
                if flatten:
                    pbar = tqdm(total=len(args))
                else:
                    loader = tqdm(loader, total=len(loader))

            for batch in loader:
                if use_forward:
                    output = forward._lf_call_transform(batch)
                else:
                    output = batch

                if use_post:
                    # TODO: unbatch not needed anymore since it is part of the post transform (Issue 19)
                    # output = unbatch(output)
                    # output = [
                    #     post._lf_call_transform(output[i])
                    #     for i in range(len(output))
                    # ]
                    output = post._lf_call_transform(output)

                if flatten:
                    # TODO Should we put the unbatch step inside of _yield_flatten_data?
                    # if not use_post:
                    #     output = unbatch(output)
                    yield from self._yield_flatten_data(output, verbose, pbar)
                    continue

                yield output

    @property
    def lf_device(self):
        """Return the device"""
        return self._lf_device

    @property
    def lf_mapdevice(self):
        """Return the map device"""
        return self._lf_mapdevice

    @property
    def lf_preprocess(self):
        """The preprocessing part (everything that happens before sending to gpu). """
        if 'cpu' in self.lf_mapdevice:
            return self
        return Identity()

    def _lf_forward_part(self):
        """The forward (GPU) part of the transform """
        if 'gpu' in self.lf_mapdevice:
            return self
        return Identity()

    @property
    def lf_forward(self):
        """The forward (GPU) part of the transform and send to GPU"""
        f = self._lf_forward_part()
        f.lf_to(self.lf_device)
        return f

    @property
    def lf_postprocess(self):
        """The postprocessing part of the transform. """
        if 'bcpu' in self.lf_mapdevice:
            return self
        return Identity()

    # DANGER: makes it mutable
    def lf_to(self, device: str):
        """Set the transform's device to *device*.

        :param device: device on which to map {'cpu', 'cuda'}
        """
        self._lf_device = device
        for item in self.__dict__:
            obj_ = self._lf_getattribute_object(item)
            if isinstance(obj_, Transform):
                obj_.lf_to(device)
            elif isinstance(obj_, list) and obj_ and isinstance(obj_[0], Transform):
                for a_trans in obj_:
                    a_trans.lf_to(device)
        return self

    def _lf_getattribute_object(self, item):
        """Like getattribute, but not returning variable values, but variable objects. """
        return object.__getattribute__(self, item)

    @property
    def lf_is_identity(self):
        """Return *True* iff the transform is the identity transform. """
        return False

    @property
    def lf_layers(self):
        """Get a dict with all layers in the transform (including layers in sub-transforms)."""
        if self._lf_layers is None:
            layer_dict = {}
            for item in self.__dict__:
                attrib = self._lf_getattribute_object(item)
                if isinstance(attrib, Transform):
                    layer_dict.update(attrib.lf_layers)
                elif type(attrib) in {tuple, list} and attrib and isinstance(attrib[0], Transform):
                    for y in attrib:
                        layer_dict.update(y.lf_layers)
            self._lf_layers = layer_dict
        return self._lf_layers

    @property
    def lf_stage(self):
        """
        :return: Stage of Transform
        """
        return self._lf_stage

    @contextlib.contextmanager
    def lf_set_stage(self, stage: str):
        """Set of stage of Transform

        :param stage: stage ('train', 'eval', 'infer')
        """
        assert stage in ('train', 'eval', 'infer')

        layers = self.lf_layers
        try:
            for layer in layers.values():
                if stage == 'train':
                    layer.train()
                else:
                    layer.eval()
            Transform._lf_stage = stage
            yield
        # TODO: Should we put layers in eval mode by default?
        finally:
            for layer in layers.values():
                layer.eval()
            Transform._lf_stage = None

    @staticmethod
    def _lf_get_loader(iterator, loader_kwargs=None):
        """Get the data loader

        :param iterator: Iterator
        :param loader_kwargs: key word arguments for the data loader
        """
        loader = DataLoader(
            iterator,
            worker_init_fn=lambda _: np.random.seed(),
            **loader_kwargs
        )
        return loader

    def infer_apply(self, arg):
        """Call transform within the infer context"""
        with self.lf_set_stage('infer'), torch.no_grad():
            return self._lf_call_transform(arg)

    def eval_apply(self, arg, loader_kwargs=None, verbose=False, flatten=False):
        """Call transform within the eval context"""
        return self._lf_callyield(arg, 'eval', loader_kwargs=loader_kwargs,
                                  verbose=verbose, flatten=flatten)

    def train_apply(self, arg, loader_kwargs=None, verbose=False, flatten=False):
        """Call transform within the train context"""
        return self._lf_callyield(arg, 'train', loader_kwargs=loader_kwargs,
                                  verbose=verbose, flatten=flatten)


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms), in contrast to `CompoundTransform`s.

    :param call:
    :param call_info:
    :param name: name of the transform
    """

    def __init__(self, call, call_info, name=None):
        super().__init__(call_info, name)
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
    """Function Transform

    :param function: function to call
    :param call_info:
    :param name: name of the transform
    :param call:
    """
    def __init__(self, function, call_info, name=None, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call, call_info, name)
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


class CompoundTransform(Transform):
    """Abstract base class for compound-transforms (transforms combining other transforms).

    :param transforms: list of transforms
    :param call_info:
    :param name: name of CompoundTransform
    :param lf_group:
    """
    op = NotImplemented

    def __init__(self, transforms, call_info, name=None, lf_group=False):
        super().__init__(call_info, name)

        self._lf_group = True if name is not None else lf_group

        self._lf_preprocess = None
        self._lf_forward = None
        self._lf_postprocess = None

        transforms = self._flatten_list(transforms)
        self.transforms = transforms

        # self._lf_mapdevice_list = None
        self._lf_mapdevice_list = [t.lf_mapdevice for t in self.transforms]
        try:
            self._mapdevice = set.union(*self._lf_mapdevice_list)
        except (AttributeError, TypeError):
            self._mapdevice = None

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
                if transform._lf_group:
                    list_flat.append(transform)
                else:
                    list_flat += transform.transforms
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

    def grouped(self):
        """Return a grouped version of *self*. """
        return type(self)(self.transforms, self._lf_call_info, name=self.lf_name, group=True)

    @staticmethod
    def _lf_get_keys(transforms):
        """Get deduplicated keys from list of transforms

        Names are updated as below.
        [None, None, 'a', 'a', 'b', 'c'] -> ['out_0', 'out_1', 'a_0', 'a_1', 'b', 'c']

        :param transforms: list of transforms
        :return: list of keys
        """
        names = []
        for ind, transform_ in enumerate(transforms):
            if transform_.lf_name is None:
                name = 'out_'+str(ind)
            else:
                name = transform_.lf_name
            names.append(name)

        counter = Counter(names)
        updated_counter = Counter()
        deduped_keys = []

        for name in names:
            new_name = name + '_' + str(updated_counter[name]) if counter[name] > 1 else name
            updated_counter.update({name: 1})
            deduped_keys.append(new_name)
        return deduped_keys


class Compose(CompoundTransform):
    """Apply series of transforms on input.

    Compose([t1, t2, t3])(x) = t3(t1(t2(x)))

    :param transforms: List of transforms to compose.
    :param flatten: If *True* flatten transforms -
        Compose([Compose([a,b]), c]) becomes Compose([a, b, c])
    :return: output from series of transforms
    """
    op = '>>'

    def __call__(self, arg):
        """Call method for Compose

        :param arg: arguments to call with
        :return: output from series of transforms
        """
        for transform_ in self.transforms:
            arg = transform_._lf_call_transform(arg)
        return arg

    # TODO Finish adding
    # def __len__(self):
    #     # TODO Previously was length of first element of trans_list. Any reason why?
    #     # return len(self.trans_list[0])
    #     return len(self.transforms)

    def _lf_list_of_forward_parts(self):
        """Accumulate all forward parts of the transforms"""
        ts_ = []
        for a_trans, dev in zip(self.transforms, self._lf_mapdevice_list):
            if 'gpu' in dev:
                # TODO Why is there a special case here?
                if len(dev) == 1:
                    ts_.append(a_trans)
                else:
                    ts_.append(a_trans.lf_forward)
        return ts_

    def _lf_forward_part(self):
        """Forward part of transforms"""
        if self._lf_forward is None:
            ts_ = self._lf_list_of_forward_parts()

            if len(ts_) == 1:
                self._lf_forward = ts_[0]
            elif ts_:
                self._lf_forward = Compose(ts_, call_info=self._lf_call_info)
            else:
                self._lf_forward = Identity()

        return self._lf_forward

    @property
    def lf_preprocess(self):
        if self._lf_preprocess is None:
            t = [
                t.lf_preprocess for t, d in zip(self.transforms, self._lf_mapdevice_list) if 'cpu' in d
            ]

            if len(t) == 1:
                self._lf_preprocess = t[0].lf_preprocess
            elif t:
                self._lf_preprocess = Compose(t, call_info=self._lf_call_info)
            else:
                self._lf_preprocess = Identity()

        return self._lf_preprocess

    @property
    def lf_postprocess(self):
        if self._lf_postprocess is None:
            t = [
                t for t, d in zip(self.transforms, self._lf_mapdevice_list) if 'bcpu' in d
            ]

            if len(t) == 1:
                self._lf_postprocess = t[0].lf_postprocess
            elif t:
                self._lf_postprocess = Compose(t, call_info=self._lf_call_info)
            else:
                self._lf_postprocess = Identity()

        return self._lf_postprocess


class Rollout(CompoundTransform):
    """Apply a list of transform to same input and get tuple output

    Rollout([t1, t2, ...])(x) := (t1(x), t2(x), ...)

    :param transforms: List of transforms to rollout.
    :param flatten: If *True* flatten transforms -
        Rollout([Rollout([a,b]), c]) becomes Rollout([a, b, c])
    :return: namedtuple of outputs
    """
    op = '+'

    def __init__(self, transforms, call_info, name=None, lf_group=False):
        super().__init__(transforms, call_info, name=name, lf_group=lf_group)
        self.lf_keys = self._lf_get_keys(self.transforms)
        self._lf_output_format = namedtuple('namedtuple', self.lf_keys)

    def __call__(self, arg):
        """Call method for Rollout

        :param arg: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for transform_ in self.transforms:
            out.append(transform_._lf_call_transform(arg))
        out = self._lf_output_format(*out)
        return out

    # TODO Finish adding
    # def __len__(self):
    #     lens = [len(x) for x in self.transforms]
    #     assert len(set(lens)) == 1
    #     return lens[0]

    @property
    def lf_preprocess(self):
        if self._lf_preprocess is None:
            t_list = [x.lf_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_preprocess = Identity()
            elif len(list(self._lf_mapdevice)) >= 2 and 'bcpu' in self._mapdevice:
                self._lf_preprocess = Parallel(t_list, call_info=self._lf_call_info)
            else:
                self._lf_preprocess = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_preprocess

    def _lf_forward_part(self):
        """Forward part"""
        # TODO Should we set self._lf_forward to Identity() like in lf_preprocess and lf_postprocess
        if self._lf_forward is None:
            t_list = [x.lf_forward for x in self.transforms]
            if len(list(self._mapdevice)) >= 2 and 'gpu' in self._lf_mapdevice:
                self._lf_forward = Parallel(t_list, call_info=self._lf_call_info)
            else:
                self._lf_forward = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_forward

    @property
    def lf_postprocess(self):
        """Post process part"""
        if self._lf_postprocess is None:
            t_list = [x.lf_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_postprocess = Identity()
            elif len(list(self._lf_mapdevice)) >= 2 and 'bcpu' in self._mapdevice:
                self._lf_postprocess = Parallel(t_list, call_info=self._lf_call_info)
            else:
                self._lf_postprocess = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_postprocess


class Parallel(CompoundTransform):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param flatten: If *True* flatten transforms -
        Parallel([Parallel([a,b]), c]) becomes Parallel([a, b, c])
    :return: namedtuple of outputs
    """
    op = '/'

    def __init__(self, transforms, call_info, name=None, lf_group=False):
        super().__init__(transforms, call_info=call_info, name=name, lf_group=lf_group)
        self.lf_keys = self._lf_get_keys(self.transforms)
        self._lf_output_format = namedtuple('namedtuple', self.lf_keys)

    def __call__(self, arg):
        """Call method for Parallel

        :param arg: Argument to call with
        :return: namedtuple of output
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            out.append(transform_._lf_call_transform(arg[ind]))
        out = self._lf_output_format(*out)
        return out

    @property
    def lf_preprocess(self):
        if self._lf_preprocess is None:
            t_list = [x.lf_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_preprocess = Identity()
            else:
                self._lf_preprocess = Parallel(t_list, call_info=self._lf_call_info)
        return self._lf_preprocess

    @property
    def lf_postprocess(self):
        if self._lf_postprocess is None:
            t_list = [x.lf_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_postprocess = Identity()
            else:
                self._lf_postprocess = Parallel(t_list, call_info=self._lf_call_info)
        return self._lf_postprocess

    def _lf_forward_part(self):
        if self._lf_forward is None:
            t_list = [x.lf_forward for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_forward = Identity()
            else:
                self._lf_forward = Parallel(t_list, call_info=self._lf_call_info)
        return self._lf_forward


class Identity(Transform):
    """Do nothing.

    :param name: name of the transform
    """

    def __init__(self, name=None):
        super().__init__(None, name=name)

    @property
    def lf_is_identity(self):
        return True

    def __call__(self, *args):
        # remove, make consistent
        if len(args) == 1:
            return args[0]
        return args


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


def group(transform: Union[Rollout, Parallel]):
    return transform.grouped()
