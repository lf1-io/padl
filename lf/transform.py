"""The Transform class and some of its children. """
import ast
from copy import copy
from collections import Counter, namedtuple
import contextlib
import inspect
from itertools import chain
from pathlib import Path
import types
from typing import Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lf.data import SimpleIterator
from lf.dumptools import var2mod, thingfinder, inspector


class _Notset:
    # pylint: disable=too-few-public-methods
    pass


_notset = _Notset()


def _isinstance_of_namedtuple(arg):
    """Check if input *arg* is instance of namedtuple"""
    typ = type(arg)
    base = typ.__bases__
    if len(base) != 1 or base[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(field, str) for field in fields)


Stage = Literal['infer', 'eval', 'train']
Component = Literal['preprocess', 'forward', 'postprocess']


class Transform:
    """Transform base class.

    :param call_info:
    :param lf_name: name of the transform
    """
    lf_stage = None

    def __init__(self, call_info: Optional[inspector.CallInfo] = None,
                 lf_name: Optional[str] = None):
        if call_info is None:
            call_info = inspector.CallInfo()
        self._lf_call_info = call_info
        self._lf_varname = _notset
        self._lf_name = lf_name
        self._lf_component = {'forward'}
        self._lf_device = 'gpu'
        self._lf_layers = None

    @property
    def lf_name(self) -> Optional[str]:
        """The "name" of the transform.

        A transform can have a name. This is optional, but helps when inspecting complex transforms.
        Good transform names indicate what the transform does.

        If a transform does not have an explicitly set name, the name will default to the name of
        the *last variable the transforms was assigned to*.
        """
        if self._lf_name is None:
            return self.lf_varname()
        return self._lf_name

    def __rshift__(self, other: "Transform") -> "Compose":
        """Compose with *other*.

        Example:
            t = a >> b >> c
        """
        return Compose([self, other])

    def __add__(self, other: "Transform") -> "Rollout":
        """ Rollout with *other*.

        Example:
            t = a + b + c
        """
        return Rollout([self, other])

    def __truediv__(self, other: "Transform") -> "Parallel":
        """ Parallel with *other*.

        Example:
            t = a / b / c
        """
        return Parallel([self, other])

    def __sub__(self, name: str) -> "Transform":
        """Create a named clone of the transform.

        Example:
            named_t = t - 'rescale image'
        """
        named_copy = copy(self)
        named_copy._lf_name = name
        return named_copy

    def lf_pre_save(self, path: Path, i: int):
        """Method that is called on each transform before saving.

        :param path: The save path.
        :param i: Subtransform index.
        """

    def lf_post_load(self, path: Path, i: int):
        """Method that is called on each transform after loading.

        :param path: The load path.
        :param i: Subtransform index.
        """

    def lf_save(self, path: Union[Path, str]):
        """Save the transform to *path*. """
        path = Path(path)
        path.mkdir(exist_ok=True)
        for i, subtrans in enumerate(self.lf_all_transforms()):
            subtrans.lf_pre_save(path, i)
        with open(path / 'transform.py', 'w') as f:
            f.write(self.lf_dumps())

    def _lf_codegraph_startnode(self, name: str) -> var2mod.CodeNode:
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

    @property
    def _lf_closurevars(self) -> Tuple[dict, dict]:
        """Return the closurevars (globals and nonlocals) the transform depends on. """
        return {}, {}

    def _lf_build_codegraph(self, graph: Optional[dict] = None,
                            scopemap: Optional[dict] = None,
                            name: Optional[str] = None,
                            scope: Optional[thingfinder.Scope] = None) -> Tuple[dict, dict]:
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

        globals_dict, nonlocals_dict = self._lf_closurevars
        all_vars_dict = {**globals_dict, **nonlocals_dict}

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

    def lf_evaluable_repr(self, indent: int = 0, var_transforms: Optional[dict] = None) -> str:  # TODO: var_transforms needed?
        # pylint: disable=unused-argument,no-self-use
        """Return a string that if evaluated *in the same scope where the transform was created*
        creates the transform. """
        return NotImplemented

    def lf_all_transforms(self):
        """Return a list of all transforms needed for executing the transform.

        This includes the transform itself, the subtransforms of a compount transform or
        transforms a function-transform depends on as a global. """
        # pylint: disable=no-self-use
        return NotImplemented

    def lf_dumps(self) -> str:
        """Dump the transform as python code. """
        scope = thingfinder.Scope.toplevel(inspector.caller_module())
        graph, scopemap = self._lf_build_codegraph(name='_lf_main', scope=scope)
        unscoped = var2mod.unscope_graph(graph, scopemap)
        return var2mod.dumps_graph(unscoped)

    def lf_repr(self, indent: int = 0) -> str:
        # pylint: disable=unused-argument
        varname = self.lf_varname()
        evaluable_repr = self.lf_evaluable_repr()
        if varname is None or varname == evaluable_repr:
            return f'{evaluable_repr}'
        return f'{evaluable_repr} [{varname}]'

    def __repr__(self) -> str:
        return self.lf_repr()

    def _lf_find_varname(self, scopedict: dict) -> Optional[str]:
        """Find the name of the variable name the transform was last assigned to.

        :returns: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        try:
            return [
                k for k, v in scopedict.items()
                if v is self and not k.startswith('_')
            ][0]
        except IndexError:
            return None

    def lf_varname(self) -> Optional[str]:
        """The name of the variable name the transform was last assigned to.

        Example:

        >>> foo = MyTransform()
        >>> foo._lf_varname
        "foo"

        :returns: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        if self._lf_varname is _notset:
            self._lf_varname = self._lf_find_varname(self._lf_call_info.module.__dict__)
        return self._lf_varname

    def _lf_set_varname(self, val):  # TODO: needed (used in wrap, but can be done without "set" method potentially)
        self._lf_varname = val

    def _lf_call_transform(self, arg, stage: Optional[Stage] = None):
        """Call transform with possibility to pass multiple arguments"""

        if stage in ('eval', 'infer'):  # TODO: move to lf_set_stage?
            torch_context = torch.no_grad()
        else:
            torch_context = contextlib.suppress()

        signature_parameters = inspect.signature(self).parameters

        with self.lf_set_stage(stage), torch_context:
            if len(signature_parameters) == 1:
                return self(arg)
            return self(*arg)

    def _lf_callyield(self, args, stage: Stage, loader_kwargs: Optional[dict] = None,
                      verbose: bool = False, flatten: bool = False):  # TODO: different name?
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param stage: Stage to call in ("eval", "train" or "infer")
        :param loader_kwargs: Data loader keyword arguments.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.

        :returns: A generator that allows iterating over the output.
        """
        assert stage in ('eval', 'train'), '_lf_callyield can only be used with stage eval or train'

        preprocess = self.lf_preprocess
        forward = self.lf_forward
        post = self.lf_postprocess

        use_preprocess = not preprocess.lf_is_identity
        use_forward = not forward.lf_is_identity
        use_post = not post.lf_is_identity

        if use_preprocess:
            iterator = SimpleIterator(
                args,
                lambda *args: self.lf_preprocess._lf_call_transform(*args, stage)
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
                output = forward._lf_call_transform(batch, stage)
            else:
                output = batch

            if use_post:
                output = post._lf_call_transform(output, stage)

            if flatten:
                if not use_post:
                    output = Unbatchify()(batch)
                yield from self._lf_yield_flatten_data(output, verbose, pbar)
                continue

            yield output

    @staticmethod
    def _lf_yield_flatten_data(batch, verbose, pbar):
        """Yield from the flattened data.

        :param batch: unbatched data type
        :param verbose: If *true* show a progress bar.
        :param pbar: tqdm progress bar
        """
        for datapoint in batch:
            yield datapoint
            if verbose:
                pbar.update(1)

    @property
    def lf_device(self) -> str:  # TODO: remove?
        """Return the device the transform is on."""
        return self._lf_device

    @property
    def lf_component(self) -> Set[Component]:
        """Return the component (preprocess, forward or postprocess)."""
        return self._lf_component

    @property
    def lf_preprocess(self) -> "Transform":
        """The preprocessing part. """
        if 'preprocess' in self.lf_component:
            return self
        return Identity()

    def _lf_forward_part(self) -> "Transform":
        """The forward part of the transform """
        if 'forward' in self.lf_component:
            return self
        return Identity()

    @property
    def lf_forward(self) -> "Transform":
        """The forward part of the transform and send to GPU"""
        f = self._lf_forward_part()
        f.lf_to(self.lf_device)  # TODO: do this?
        return f

    @property
    def lf_postprocess(self) -> "Transform":
        """The postprocessing part of the transform. """
        if 'postprocess' in self.lf_component:
            return self
        return Identity()

    def lf_to(self, device: str):  # TODO: change
        """Set the transform's device to *device*.

        :param device: device on which to map {'cpu', 'gpu'}
        """
        self._lf_device = device
        return self

    @property
    def lf_is_identity(self):  # TODO: keep?
        """Return *True* iff the transform is the identity transform. """
        return False

    @property
    def lf_layers(self) -> List[torch.nn.Module]:
        """Get a dict with all layers in the transform (including layers in sub-transforms)."""
        layers = []
        for subtrans in self.lf_all_transforms():
            if isinstance(subtrans, torch.nn.Module):
                layers.append(subtrans)
        return layers

    @contextlib.contextmanager
    def lf_set_stage(self, stage: Optional[str]=None):
        """Set of stage of Transform

        :param stage: stage ('train', 'eval', 'infer')
        """
        assert stage in ('train', 'eval', 'infer', None)

        if stage is None:
            yield
            return

        layers = self.lf_layers  # TODO: set back?
        try:
            for layer in layers:
                if stage == 'train':
                    layer.train()
                else:
                    layer.eval()
            Transform.lf_stage = stage
            yield
        # TODO: Should we put layers in eval mode by default?
        finally:
            for layer in layers:
                layer.eval()
            Transform.lf_stage = None

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

    def infer_apply(self, args):  # TODO: *arg? **kwarg?
        """Call transform within the infer context.

        This expects a single argument and returns a single output.
        """
        return self._lf_call_transform(args, stage='infer')

    def eval_apply(self, args: Iterable, loader_kwargs: Optional[dict] = None,
                   verbose: bool = False, flatten: bool = False):
        """Call transform within the eval context.

        This expects an iterable input and returns a generator.

        :param args: The arguments - an iterable (e.g. list) of inputs.
        :param loader_kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._lf_callyield(args, 'eval', loader_kwargs=loader_kwargs,
                                  verbose=verbose, flatten=flatten)

    def train_apply(self, args: Iterable, loader_kwargs: Optional[dict] = None,
                    verbose: bool = False, flatten: bool = False):
        """Call transform within the train context.

        This expects an iterable input and returns a generator.

        :param args: The arguments - an iterable (e.g. list) of inputs.
        :param loader_kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._lf_callyield(args, 'train', loader_kwargs=loader_kwargs,
                                  verbose=verbose, flatten=flatten)


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms - in contrast to `CompoundTransform`s).

    Examples of `AtomicTransform`s are `ClassTransform`s and `FunctionTransform`s.

    :param call:
    :param call_info:
    :param lf_name: name of the transform
    """

    def __init__(self, call: str, call_info: Optional[inspector.CallInfo] = None,
                 lf_name: Optional[str] = None):
        super().__init__(call_info, lf_name)
        self._lf_call = call

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        if var_transforms is None:
            var_transforms = {}
        return var_transforms.get(self, self._lf_call)

    def lf_all_transforms(self):
        res = [self]
        globals_dict, nonlocals_dict = self._lf_closurevars
        for v in chain(self.__dict__.values(), globals_dict.values(), nonlocals_dict.values()):
            try:
                children = v.lf_all_transforms()
            except AttributeError:
                continue
            for child_transform in children:
                if child_transform not in res:
                    res.append(child_transform)
        return res


class FunctionTransform(AtomicTransform):
    """Function Transform

    :param function: function to call
    :param call_info:
    :param lf_name: name of the transform
    :param call:
    """
    def __init__(self, function, call_info, lf_name=None, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call=call, call_info=call_info, lf_name=lf_name)
        self.function = function

    @property
    def _lf_closurevars(self) -> inspect.ClosureVars:
        closurevars = inspect.getclosurevars(self.function)
        return closurevars.globals, closurevars.nonlocals

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ClassTransform(AtomicTransform):
    """Class Transform

    :param lf_name: name of the transform
    """
    def __init__(self, lf_name=None):
        caller_frameinfo = inspector.non_init_caller_frameinfo()
        call_info = inspector.CallInfo(caller_frameinfo)
        call = inspector.get_call_segment_from_frame(caller_frameinfo.frame)
        AtomicTransform.__init__(
            self,
            call=call,
            call_info=call_info,
            lf_name=lf_name
        )


class TorchModuleTransform(ClassTransform):
    """Torch Module Transform"""

    def lf_pre_save(self, path, i):
        """
        :param path: The save path.
        :param i: Sublayer index.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def lf_post_load(self, path, i):
        """
        :param path: The load path.
        :param i: Sublayer index.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))


class CompoundTransform(Transform):
    """Abstract base class for compound-transforms (transforms combining other transforms).

    :param transforms: list of transforms
    :param call_info:
    :param lf_name: name of CompoundTransform
    :param lf_group:
    """
    op = NotImplemented

    def __init__(self, transforms, call_info=None, lf_name=None, lf_group=False):
        super().__init__(call_info, lf_name)

        self._lf_group = True if lf_name is not None else lf_group
        self._lf_preprocess = None
        self._lf_forward = None
        self._lf_postprocess = None

        transforms = self._flatten_list(transforms)
        self.transforms = transforms

        # self._lf_component_list = None
        self._lf_component_list = [t.lf_component for t in self.transforms]
        try:
            self._lf_component = set.union(*self._lf_component_list)
        except (AttributeError, TypeError):
            self._lf_component = None

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        sub_reprs = [
            x.lf_varname() or x.lf_evaluable_repr(indent + 4, var_transforms)
            for x in self.transforms
        ]
        result = (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )
        if self._lf_group:
            result = 'group' + result
        return result

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

        :param transform_list: List of transforms.
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
        return type(self)(self.transforms, self._lf_call_info, lf_name=self.lf_name, lf_group=True)

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
    :param call_info:
    :param lf_name: name of the Compose transform
    :param lf_group:
    :return: output from series of transforms
    """
    op = '>>'

    def __init__(self, transforms, call_info=None, lf_name=None, lf_group=False):
        super().__init__(transforms, call_info=call_info, lf_name=lf_name, lf_group=lf_group)

        preprocess_end = 0
        postprocess_start = len(self.transforms)
        set_postprocess = True
        for i, transform_ in enumerate(self.transforms):
            if 'preprocess' in transform_.lf_component:
                preprocess_end = i
            if 'postprocess' in transform_.lf_component and set_postprocess:
                postprocess_start = i
                set_postprocess = False
        for i in range(preprocess_end):
            self._lf_component_list[i] = {'preprocess'}
        for i in range(postprocess_start+1, len(self.transforms)):
            self._lf_component_list[i] = {'postprocess'}

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
        for transform_, component in zip(self.transforms, self._lf_component_list):
            if 'forward' in component:
                # TODO I don't think special case is needed anymore
                if len(component) == 1:
                    ts_.append(transform_)
                else:
                    ts_.append(transform_.lf_forward)
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
            t_list = [
                t for t, comp in zip(self.transforms, self._lf_component_list) if 'preprocess' in comp
            ]

            if len(t_list) == 1:
                # TODO Test this line, I think this will cause an error
                self._lf_preprocess = t_list[0].lf_preprocess
            elif t_list:
                self._lf_preprocess = Compose(t_list, call_info=self._lf_call_info)
            else:
                self._lf_preprocess = Identity()

        return self._lf_preprocess

    @property
    def lf_postprocess(self):
        if self._lf_postprocess is None:
            t_list = [
                t for t, comp in zip(self.transforms, self._lf_component_list) if 'postprocess' in comp
            ]

            if len(t_list) == 1:
                # TODO Test this line, I think this will cause an error
                self._lf_postprocess = t_list[0].lf_postprocess
            elif t_list:
                self._lf_postprocess = Compose(t_list, call_info=self._lf_call_info)
            else:
                self._lf_postprocess = Identity()

        return self._lf_postprocess


class Rollout(CompoundTransform):
    """Apply a list of transform to same input and get tuple output

    Rollout([t1, t2, ...])(x) := (t1(x), t2(x), ...)

    :param transforms: List of transforms to rollout.
    :param call_info:
    :param lf_name: name of the Rollout transform
    :param lf_group:
    :return: namedtuple of outputs
    """
    op = '+'

    def __init__(self, transforms, call_info=None, lf_name=None, lf_group=False):
        super().__init__(transforms, call_info=call_info, lf_name=lf_name, lf_group=lf_group)
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
            else:
                self._lf_preprocess = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_preprocess

    def _lf_forward_part(self):
        # TODO Should we set self._lf_forward to Identity() like in lf_preprocess and lf_postprocess
        if self._lf_forward is None:
            t_list = [x.lf_forward for x in self.transforms]
            if len(list(self._lf_component)) >= 2 and 'forward' in self._lf_component:
                self._lf_forward = Parallel(t_list, call_info=self._lf_call_info)
            else:
                self._lf_forward = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_forward

    @property
    def lf_postprocess(self):
        if self._lf_postprocess is None:
            t_list = [x.lf_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_postprocess = Identity()
            elif len(list(self._lf_component)) >= 2 and 'postprocess' in self._lf_component:
                self._lf_postprocess = Parallel(t_list, call_info=self._lf_call_info)
            else:
                self._lf_postprocess = Rollout(t_list, call_info=self._lf_call_info)
        return self._lf_postprocess


class Parallel(CompoundTransform):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param call_info:
    :param lf_name: name of the Parallel transform
    :param lf_group:
    :return: namedtuple of outputs
    """
    op = '/'

    def __init__(self, transforms, call_info=None, lf_name=None, lf_group=False):
        super().__init__(transforms, call_info=call_info, lf_name=lf_name, lf_group=lf_group)
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

    def _lf_forward_part(self):
        if self._lf_forward is None:
            t_list = [x.lf_forward for x in self.transforms]
            self._lf_forward = Parallel(t_list, call_info=self._lf_call_info)
        return self._lf_forward

    @property
    def lf_postprocess(self):
        if self._lf_postprocess is None:
            t_list = [x.lf_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._lf_postprocess = Identity()
            else:
                self._lf_postprocess = Parallel(t_list, call_info=self._lf_call_info)
        return self._lf_postprocess


class Identity(ClassTransform):
    """Do nothing.

    :param lf_name: name of the transform
    """

    def __init__(self, lf_name=None):
        super().__init__(lf_name=lf_name)

    @property
    def lf_is_identity(self):
        return True

    def __call__(self, args):
        return args


class Unbatchify(ClassTransform):
    """Remove batch dimension (inverse of Batchify).

    :param dim: batching dimension
    """

    def __init__(self, dim=0, lf_name=None):
        super().__init__(lf_name=lf_name)
        self.dim = dim
        self._lf_component = {'postprocess'}

    def __call__(self, args):
        assert Transform.lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if Transform.lf_stage != 'infer':
            return args
        if isinstance(args, (tuple, type(namedtuple))):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.squeeze(self.dim)

        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(ClassTransform):
    """Add a batch dimension at dimension *dim*. During inference, this unsqueezes
    tensors and, recursively, tuples thereof.

    :param dim: batching dimension
    """

    def __init__(self, dim=0, lf_name=None):
        super().__init__(lf_name=lf_name)
        self.dim = dim
        self._lf_component = {'preprocess'}

    def __call__(self, args):
        assert Transform.lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if Transform.lf_stage != 'infer':
            return args
        if isinstance(args, (tuple, list, type(namedtuple))):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim)
        if isinstance(args, (float, int)):
            return torch.tensor([args])
        raise TypeError('only tensors and tuples of tensors recursively supported...')


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
    for i, subtrans in enumerate(transform.lf_all_transforms()):
        subtrans.lf_post_load(path, i)
    return transform


def group(transform: Union[Rollout, Parallel]):
    return transform.grouped()
