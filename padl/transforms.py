"""The Transform class and its fundamental children.

Transforms should be created using the `padl.transform` wrap-function.
"""
import ast
import re
from copy import copy
from collections import Counter, namedtuple, OrderedDict
import contextlib
import inspect
from itertools import chain
from pathlib import Path
from shutil import rmtree
import textwrap
import types
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from padl.data import SimpleDataset
from padl.dumptools import var2mod, symfinder, inspector
from padl.dumptools.serialize import Serializer

from padl.dumptools.packagefinder import dump_packages_versions
from padl.exceptions import WrongDeviceError
from padl.print_utils import combine_multi_line_strings, create_reverse_arrow, make_bold, \
    make_green, create_arrow, format_argument, visible_len


class _Notset:
    # pylint: disable=too-few-public-methods
    def __bool__(self):
        return False


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


def _move_to_device(args, device):
    """Move args to given device

    :param args: args to move to device
    :param device: device to move to
    """
    if isinstance(args, (tuple, list)):
        return tuple([_move_to_device(x, device) for x in args])
    if isinstance(args, torch.Tensor):
        return args.to(device)
    return args


Stage = Literal['infer', 'eval', 'train']
Component = Literal['preprocess', 'forward', 'postprocess']


class Transform:
    """Transform base class.

    :param call_info: A `CallInfo` object containing information about the how the transform was
    created (needed for saving).
    :param pd_name: name of the transform
    """
    pd_stage = None

    def __init__(self, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        if call_info is None:
            call_info = inspector.CallInfo()
        self._pd_call_info = call_info
        self._pd_varname = _notset
        self._pd_name = pd_name
        self._pd_component = {'forward'}
        self._pd_device = 'cpu'
        self._pd_layers = None

    @property
    def pd_name(self) -> Optional[str]:
        """The "name" of the transform.

        A transform can have a name. This is optional, but helps when inspecting complex transforms.
        Good transform names indicate what the transform does.

        If a transform does not have an explicitly set name, the name will default to the name of
        the *last variable the transforms was assigned to*.
        """
        if not self._pd_name:
            return self.pd_varname()
        return self._pd_name

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
        named_copy._pd_name = name
        named_copy._pd_varname = _notset
        return named_copy

    def __invert__(self) -> "Map":
        """Map.

        Example:
            t = ~a
        """
        return Map(self)

    def pd_pre_save(self, path: Path, i: int):
        """Method that is called on each transform before saving.

        This normally does nothing. Override to implement custom serialization.

        :param path: The save-folder path.
        :param i: Unique transform index, can be used to construct filenames.
        """

    def pd_post_load(self, path: Path, i: int):
        """Method that is called on each transform after loading.

        This normally does nothing. Override to implement custom serialization.

        :param path: The load path.
        :param i: Unique transform index, can be used to construct filenames.
        """

    def pd_save(self, path: Union[Path, str], force_overwrite: bool = False):
        """Save the transform to a folder at *path*.

        The folder's name should end with '.padl'. If no extension is given, it will be added
        automatically.

        If the folder exist, call with *force_overwrite* = `True` to overwrite. Otherwise, this
        will raise a FileExistsError.
        """
        path = Path(path)
        if path.suffix == '':
            path = path.parent / (path.name + '.padl')

        if path.exists():
            if not force_overwrite:
                raise FileExistsError(f'{path} exists, call with *force_overwrite* to overwrite.')
            rmtree(path)

        path.mkdir()

        for i, subtrans in enumerate(self._pd_all_transforms()):
            subtrans.pd_pre_save(path, i)
        code, versions = self._pd_dumps(True, path=path)

        with open(path / 'transform.py', 'w') as f:
            f.write(code)
        with open(path / 'versions.txt', 'w') as f:
            f.write(versions)

    def _pd_codegraph_startnode(self, name: str) -> var2mod.CodeNode:
        """Build the start-code-node - the node with the source needed to create *self* as "name".
        (in the scope where *self* was originally created). """
        start_source = f'{name or "_pd_dummy"} = {self._pd_evaluable_repr()}'
        start_node = ast.parse(start_source).body[0]
        start_globals = {
            (var, self._pd_call_info.scope)  # this should be the current scope ...?
            for var in var2mod.find_globals(start_node)
        }
        return var2mod.CodeNode(
            source=start_source,
            ast_node=start_node,
            globals_=start_globals
        )

    @property
    def _pd_closurevars(self) -> Tuple[dict, dict]:
        """Return the closurevars (globals and nonlocals) the transform depends on. """
        return {}, {}

    def _pd_build_codegraph(self, graph: Optional[dict] = None,
                            scopemap: Optional[dict] = None,
                            name: Optional[str] = None,
                            scope: Optional[symfinder.Scope] = None) -> Tuple[dict, dict]:
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        if scope is None:
            scope = self._pd_call_info.scope

        given_name = name

        try:
            if self._pd_call == name:
                name = None
        except AttributeError:
            pass

        # build the start node ->
        start = self._pd_codegraph_startnode(name)
        # <-

        globals_dict, nonlocals_dict = self._pd_closurevars
        all_vars_dict = {**globals_dict, **nonlocals_dict}

        # find dependencies
        todo = {*start.globals_}
        while todo and (next_ := todo.pop()):
            # we know this already - go on
            if next_ in scopemap:
                continue

            next_var, next_scope = next_

            if next_var.startswith('PADL_VALUE'):
                continue

            # see if the object itself knows how to generate its codegraph
            try:
                if len(next_scope) > 0:
                    next_obj = all_vars_dict[next_var]
                else:
                    next_obj = globals_dict[next_var]
                next_obj._pd_build_codegraph(graph, scopemap, next_var)
            except (KeyError, AttributeError):
                pass
            else:
                continue

            # find how next_var came into being
            (source, node), scope_of_next_var = symfinder.find_in_scope(next_var, next_scope)
            scopemap[next_var, next_scope] = scope_of_next_var

            # find dependencies
            globals_ = {
                (var, scope_of_next_var)
                for var in var2mod.find_globals(node)
            }
            graph[next_var, scope_of_next_var] = var2mod.CodeNode(source=source, globals_=globals_,
                                                                  ast_node=node)
            todo.update(globals_)
        # find dependencies done

        if name is not None:
            assert scope is not None
            graph[name, scope] = start

        if given_name is not None:
            scopemap[given_name, scope] = scope

        return graph, scopemap

    def _pd_evaluable_repr(self, indent: int = 0) -> str:
        """Return a string that if evaluated *in the same scope where the transform was created*
        creates the transform. """
        result = self._pd_evaluable_repr_inner(indent)
        if self._pd_name is not None:
            return f"({result} - '{self._pd_name}')"
        return result

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        # pylint: disable=unused-argument,no-self-use
        """Return a string that if evaluated *in the same scope where the transform was created*
        creates the transform. """
        raise NotImplementedError

    def _pd_all_transforms(self, result: Optional[list] = None) -> list:
        """Return a list of all transforms needed for executing the transform.

        This includes the transform itself, the subtransforms of a compount transform or
        transforms a function-transform depends on as a global. """
        if result is None:
            result = []
        if self in result:
            return result
        result.append(self)
        for transform in self._pd_direct_subtransforms:
            # pylint: disable=protected-access
            transform._pd_all_transforms(result)
        return result

    @property
    def _pd_direct_subtransforms(self) -> Iterator['Transform']:
        """Iterator over the direct subtransforms. """
        # pylint: disable=no-self-use
        raise NotImplementedError

    def _pd_dumps(self, return_versions: bool = False,
                  path: Optional[Path] = None) -> Union[str, Tuple[str, str]]:
        """Dump the transform as python code.

        :param return_versions: If *True* return a tuple of the code and a file listing
            dependencies and their versions.
        :param path: Optional path to save at, might be required for serializer code snippets.
        """
        scope = symfinder.Scope.toplevel(inspector.caller_module())
        graph, scopemap = self._pd_build_codegraph(name='_pd_main', scope=scope)
        Serializer.save_all(graph, scopemap, path)
        unscoped = var2mod.unscope_graph(graph, scopemap)
        code = var2mod.dumps_graph(unscoped)
        if return_versions:
            versions = dump_packages_versions(node.ast_node for node in graph.values())
            return code, versions
        return code

    def __repr__(self):
        return self._pd_shortrepr(formatting=False)

    def _repr_pretty_(self, p, cycle) -> str:
        # pylint: disable=invalid-name
        title = self._pd_title()
        if self.pd_name is not None and self.pd_name != title:
            title = make_bold(title) + f' - "{self.pd_name}"'
        else:
            title = make_bold(title)
        top_message = title + ':' + '\n\n'
        bottom_message = textwrap.indent(self._pd_longrepr(), '   ')
        p.text(top_message + bottom_message if not cycle else '...')

    def _pd_longrepr(self, formatting=True) -> str:
        """A lone string representation of the transform."""
        raise NotImplementedError

    def _pd_parensrepr(self, formatting=True) -> str:
        short = self._pd_shortrepr(formatting)
        if len(short) < 50:
            if len(getattr(self, 'transforms', [])) > 1:
                short = f"{make_green('[', not formatting)}{short}{make_green(']', not formatting)}"
            return short
        return self._pd_tinyrepr(formatting)


    def _pd_shortrepr(self, formatting=True) -> str:
        """A short string representation of the transform."""
        return self._pd_title()

    def _pd_tinyrepr(self, formatting=True) -> str:
        """A tiny string representation of the transform."""
        return self.pd_name or f'<anonymous {self.__class__.__name__}>'

    def _pd_title(self) -> str:
        """A title for the transform."""
        return self._pd_tinyrepr()

    def _pd_find_varname(self, scopedict: dict) -> Optional[str]:
        """Find the name of the variable name the transform was last assigned to.

        :return: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        try:
            return [
                k for k, v in scopedict.items()
                if v is self and not k.startswith('_')
            ][0]
        except IndexError:
            return None

    def pd_varname(self) -> Optional[str]:
        """The name of the variable name the transform was last assigned to.

        Example:

        >>> foo = MyTransform()
        >>> foo._pd_varname
        "foo"

        :return: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        if self._pd_varname is _notset:
            module = inspector.caller_module()
            self._pd_varname = self._pd_find_varname(module.__dict__)
        return self._pd_varname

    def _pd_forward_device_check(self) -> bool:
        """Check if all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole CompoundTransform.
        """
        for layer in self.pd_forward.pd_layers:
            for parameters in layer.parameters():
                parameter_device = parameters.device.type
                if ':' in self.pd_device and 'cuda' in parameter_device:
                    parameter_device += f':{parameters.device.index}'
                if parameter_device != self.pd_device:
                    raise WrongDeviceError(self, layer)
        return True

    def _pd_unpack_argument(self, arg) -> bool:
        """Return *True* if to arguments should be unpacked, else *False*"""
        signature_count = 0
        if not isinstance(arg, (list, tuple)):
            return False

        if hasattr(self, '_pd_number_of_inputs') and self._pd_number_of_inputs is not None:
            return self._pd_number_of_inputs > 1

        try:
            parameters = self._pd_get_signature().values()
        except ValueError:
            return False
        for param in parameters:
            if param.kind in (
                    param.POSITIONAL_OR_KEYWORD,
                    param.POSITIONAL_ONLY):
                signature_count += 1
            if param.kind == param.VAR_POSITIONAL:
                return True
        if signature_count > 1:
            return True
        return False

    def _pd_call_transform(self, arg, stage: Optional[Stage] = None):
        """Call the transform, with possibility to pass multiple arguments.

        :param stage: The stage ("infer", "eval", "train") to perform the call with.
        :return: Whatever the transform returns.
        """

        if stage in ('eval', 'infer'):
            torch_context = torch.no_grad()
        else:
            torch_context = contextlib.suppress()

        with self.pd_set_stage(stage), torch_context:
            if self._pd_unpack_argument(arg):
                return self(*arg)
            return self(arg)

    def _pd_get_signature(self):
        """Get the signature of the transform. """
        return inspect.signature(self).parameters

    def _pd_itercall(self, args, stage: Stage, loader_kwargs: Optional[dict] = None,
                     verbose: bool = False, flatten: bool = False) -> Iterator:
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param stage: Stage to call in ("eval", "train" or "infer")
        :param loader_kwargs: Data loader keyword arguments.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.

        :return: A generator that allows iterating over the output.
        """
        assert stage in ('eval', 'train'), '_pd_itercall can only be used with stage eval or train'

        # self._pd_forward_device_check()

        preprocess = self.pd_preprocess
        forward = self.pd_forward
        post = self.pd_postprocess

        use_preprocess = not isinstance(preprocess, Identity)
        use_forward = not isinstance(forward, Identity)
        use_post = not isinstance(post, Identity)

        if use_preprocess:
            data = SimpleDataset(
                args,
                lambda *args: self.pd_preprocess._pd_call_transform(*args, stage),
            )
            if loader_kwargs is None:
                loader_kwargs = {}
            loader = self._pd_get_loader(sequence=data, loader_kwargs=loader_kwargs)
        else:
            loader = args

        pbar = None
        if verbose:
            if flatten:
                pbar = tqdm(total=len(args))
            else:
                loader = tqdm(loader, total=len(loader))

        for batch in loader:
            batch = _move_to_device(batch, self.pd_device)

            if use_forward:
                output = forward._pd_call_transform(batch, stage)
            else:
                output = batch

            if use_post:
                output = post._pd_call_transform(output, stage)

            if flatten:
                if verbose:
                    pbar.update()
                if not use_post:
                    output = Unbatchify(cpu=False)(batch)
                if hasattr(self, '_pd_output_format'):
                    yield from self._pd_output_format(*output)
                else:
                    yield from output
                continue
            if hasattr(self, '_pd_output_format'):
                yield self._pd_output_format(*output)
            else:
                yield output

    @property
    def pd_device(self) -> str:
        """Return the device ("cpu" / "cuda") the transform is on."""
        return self._pd_device

    @property
    def pd_component(self) -> Set[Component]:
        """Return the component (preprocess, forward or postprocess)."""
        return self._pd_component

    @property
    def pd_preprocess(self) -> "Transform":
        """The preprocessing part of the transform. """
        if 'preprocess' in self.pd_component:
            return self
        return Identity()

    @property
    def pd_forward(self) -> "Transform":
        """The forward part of the transform (that what's typically done on the GPU)."""
        if 'forward' in self.pd_component:
            return self
        return Identity()

    @property
    def pd_postprocess(self) -> "Transform":
        """The postprocessing part of the transform. """
        if 'postprocess' in self.pd_component:
            return self
        return Identity()

    def pd_to(self, device: str) -> "Transform":
        """Set the transform's device to *device*.

        :param device: Device to set the transform to {'cpu', 'cuda', 'cuda:N'}.
        """
        self._pd_device = device
        for layer in self.pd_layers:
            layer.to(device)

        self.pd_preprocess._pd_device = device
        self.pd_forward._pd_device = device
        self.pd_postprocess._pd_device = device
        return self

    @property
    def pd_layers(self) -> List[torch.nn.Module]:
        """Get a list with all pytorch layers in the transform (including layers in sub-transforms).
        """
        layers = []
        for subtrans in self._pd_all_transforms():
            if isinstance(subtrans, torch.nn.Module):
                layers.append(subtrans)
        return layers

    def pd_parameters(self) -> Iterator:
        """Iterate over all (pytorch-) parameters in all layers contained in the transform. """
        for layer in self.pd_layers:
            yield from layer.parameters()

    @contextlib.contextmanager
    def pd_set_stage(self, stage: Optional[str] = None):
        """Set of stage of Transform

        :param stage: stage ('train', 'eval', 'infer')
        """
        assert stage in ('train', 'eval', 'infer', None)

        if stage is None:
            yield
            return

        layers = self.pd_layers
        training_before = [layer.training for layer in layers]
        try:
            for layer in layers:
                if stage == 'train':
                    layer.train()
                else:
                    layer.eval()
            Transform.pd_stage = stage
            yield
        finally:
            for i, training in enumerate(training_before):
                layer = layers[i]
                if training:
                    layer.train()
                else:
                    layer.eval()
            Transform.pd_stage = None

    @staticmethod
    def _pd_get_loader(sequence, loader_kwargs=None) -> DataLoader:
        """Get a pytorch data loader.

        :param sequence: A sequence of datapoints.
        :param loader_kwargs: Keyword arguments passed to the data loader (see the pytorch
            `DataLoader` documentation for details).
        """
        return DataLoader(
            sequence,
            worker_init_fn=lambda _: np.random.seed(),
            **loader_kwargs
        )

    def infer_apply(self, inputs):
        """Call transform within the infer context.

        This expects a single argument and returns a single output.

        :param inputs: The input.
        """
        inputs = self.pd_preprocess._pd_call_transform(inputs, stage='infer')
        inputs = _move_to_device(inputs, self.pd_device)
        inputs = self.pd_forward._pd_call_transform(inputs, stage='infer')
        inputs = self.pd_postprocess._pd_call_transform(inputs, stage='infer')

        return inputs

    def eval_apply(self, inputs: Iterable,
                   verbose: bool = False, flatten: bool = False, **kwargs):
        """Call transform within the eval context.

        This will use multiprocessing for the preprocessing part via `DataLoader` and turn
        of gradients for the forward part.

        It expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_itercall(inputs, 'eval', loader_kwargs=kwargs,
                                 verbose=verbose, flatten=flatten)

    def train_apply(self, inputs: Iterable,
                    verbose: bool = False, flatten: bool = False, **kwargs):
        """Call transform within the train context.

        This will use multiprocessing for the preprocessing part via `DataLoader` and turn
        on gradients for the forward part.

        It expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_itercall(inputs, 'train', loader_kwargs=kwargs,
                                 verbose=verbose, flatten=flatten)


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms - in contrast to `CompoundTransform`s).

    Examples of `AtomicTransform`s are `ClassTransform`s and `FunctionTransform`s.

    :param call: The transform's call string.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: The transform's name.
    """

    def __init__(self, call: str, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        super().__init__(call_info, pd_name)
        self._pd_call = call

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        return self._pd_call

    def _pd_title(self) -> str:
        return self._pd_call

    @property
    def _pd_direct_subtransforms(self) -> Iterator[Transform]:
        # pylint: disable=no-self-use
        globals_dict, nonlocals_dict = self._pd_closurevars
        for v in chain(self.__dict__.values(), globals_dict.values(), nonlocals_dict.values()):
            if isinstance(v, Transform):
                yield v


class FunctionTransform(AtomicTransform):
    """A transform that wraps a *function*.

    Do not use this directly - rather, wrap a function using `padl.transform`,

    as a decorator::

        @transform
        def f(x):
            ...

    inline::

        t = transform(f)

    or with a lambda function::

        t = transform(lambda x: x + 1)

    :param function: The wrapped function.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: name of the transform
    :param call: The call string (defaults to the function's name).
    :param source: The source code (optional).
    """

    def __init__(self, function: Callable, call_info: inspector.CallInfo,
                 pd_name: Optional[str] = None, call: Optional[str] = None,
                 source: Optional[str] = None):
        if call is None:
            call = function.__name__
        super().__init__(call=call, call_info=call_info, pd_name=pd_name)
        self.function = function
        self._pd_number_of_inputs = None
        self._source = source

    @property
    def source(self) -> str:
        """The source of the wrapped function. """
        if self._source is not None:
            return self._source
        body_msg = inspect.getsource(self.function)
        body_msg = ''.join(re.split('(def )', body_msg, 1)[1:])
        return body_msg

    def _pd_get_signature(self) -> List[str]:
        if self._pd_number_of_inputs is None:
            return inspect.signature(self).parameters
        return [f'arg_{i}' for i in range(self._pd_number_of_inputs)]

    def _pd_longrepr(self, formatting=True) -> str:
        try:
            return '\n'.join(self.source.split('\n')[:30])
        except TypeError:
            return self._pd_call

    def _pd_shortrepr(self, formatting=True) -> str:
        if len(self._pd_longrepr().split('\n', 1)) == 1:
            return self._pd_longrepr(formatting)
        return super()._pd_shortrepr(formatting)

    def _pd_title(self) -> str:
        return self.function.__name__

    @property
    def _pd_closurevars(self) -> inspect.ClosureVars:
        try:
            closurevars = inspect.getclosurevars(self.function)
        except TypeError as exc:
            warn(f'Could not get closurevars ({exc}). This is usually fine as closurevars are only '
                 'needed for user defined transforms.',
                 RuntimeWarning)
            return {}, {}
        return closurevars.globals, closurevars.nonlocals

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ClassTransform(AtomicTransform):
    """Class Transform.

    Do not use this directly, instead, use the `transform` decorator to wrap a class.

    :param pd_name: name of the transform
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    :param arguments: ordered dictionary of initialization arguments to be used in printing
    """

    def __init__(self, pd_name: str = None, ignore_scope: bool = False,
                 arguments: Optional[OrderedDict] = None):
        caller_frameinfo = inspector.non_init_caller_frameinfo()
        call_info = inspector.CallInfo(caller_frameinfo, ignore_scope=ignore_scope)
        call = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call')
        call = re.sub(r'\n\s*', ' ', call)
        self._pd_arguments = arguments
        AtomicTransform.__init__(
            self,
            call=call,
            call_info=call_info,
            pd_name=pd_name
        )

    @property
    def source(self) -> str:
        """The class source code. """
        (body_msg, _), _ = symfinder.find_in_scope(self.__class__.__name__,
                                                   self._pd_call_info.scope)
        try:
            return 'class ' + body_msg.split('class ', 1)[1]
        except IndexError:
            return body_msg

    def _formatted_args(self) -> str:
        """Format the object's init arguments for printing. """
        if self._pd_arguments is None:
            return '-?-'

        args_list = []
        for key, value in self._pd_arguments.items():
            if key == 'args':
                args_list += [f'{format_argument(val)}' for val in value]
            elif key == 'kwargs':
                args_list += [f'{subkey}={format_argument(val)}' for subkey, val in value.items()]
            else:
                args_list.append(f'{key}={format_argument(value)}')
        return ', '.join(args_list)

    def _pd_longrepr(self) -> str:
        try:
            return '\n'.join(self.source.split('\n')[:30])
        except symfinder.NameNotFound:
            return self._pd_call

    def _pd_title(self) -> str:
        title = type(self).__name__
        return title + '(' + self._formatted_args() + ')'


class TorchModuleTransform(ClassTransform):
    """Transform class for use with `torch.nn.Module`."""

    def _pd_get_signature(self):
        return inspect.signature(self.forward).parameters

    def pd_pre_save(self, path: Path, i: int):
        """Dump the model's parameters to a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def pd_post_load(self, path, i):
        """Load the model's parameters form a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))

    def _pd_longrepr(self) -> str:
        return torch.nn.Module.__repr__(self)


class Map(Transform):
    """Apply one transform to each element of a list.

    >>> Map(t)([x1, x2, x3]) == [t(x1), t(x2), t(x3)]
    True

    :param transform: Transform to be applied to a list of inputs.
    :param call_info: A `CallInfo` object containing information about the how the transform was
    created (needed for saving).
    :param pd_name: name of the transform
    """

    def __init__(self, transform: Transform, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        super().__init__(call_info, pd_name)

        self.transform = transform
        self._pd_component = transform.pd_component

        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

    def __call__(self, args: Iterable):
        """
        :param args: Args list to call transforms with
        """
        return [self.transform._pd_call_transform(arg) for arg in args]

    def _pd_longrepr(self) -> str:
        return '~ ' + self.transform._pd_shortrepr()

    @property
    def _pd_direct_subtransforms(self) -> Iterator[Transform]:
        yield self.transform

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        varname = self.transform.pd_varname()
        if varname:
            return f'~{varname}'
        return f'~{self.transform._pd_evaluable_repr(indent)}'

    def _pd_build_codegraph(self, graph=None, scopemap=None, name=None, scope=None):
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        start = self._pd_codegraph_startnode(name)

        if name is not None:
            assert scope is not None
            graph[name, scope] = start
            scopemap[name, scope] = scope

        varname = self.transform.pd_varname()
        self.transform._pd_build_codegraph(graph, scopemap, varname,  self._pd_call_info.scope)
        return graph, scopemap

    @property
    def pd_preprocess(self) -> Transform:
        if self._pd_preprocess is None:
            t_pre = self.transform.pd_preprocess
            if isinstance(t_pre, Identity):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Map(transform=t_pre, call_info=self._pd_call_info)
        return self._pd_preprocess

    @property
    def pd_postprocess(self) -> Transform:
        if self._pd_postprocess is None:
            t_post = self.transform.pd_postprocess
            if isinstance(t_post, Identity):
                self._pd_postprocess = Identity()
            else:
                self._pd_postprocess = Map(transform=t_post, call_info=self._pd_call_info)
        return self._pd_postprocess

    @property
    def pd_forward(self) -> Transform:
        if self._pd_forward is None:
            t_for = self.transform.pd_forward
            if isinstance(t_for, Identity):
                self._pd_forward = Identity()
            else:
                self._pd_forward = Map(transform=t_for, call_info=self._pd_call_info)
        return self._pd_forward


class CompoundTransform(Transform):
    """Abstract base class for compound-transforms (transforms combining other transforms).

    :param transforms: list of transforms
    :param call_info: A `CallInfo` object containing information about the how the transform was
    created (needed for saving).
    :param pd_name: name of CompoundTransform
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `CompoundTransform`.
    """
    op = NotImplemented
    display_op = NotImplemented

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(call_info, pd_name)

        self._pd_group = True if pd_name is not None else pd_group

        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

        transforms = self._flatten_list(transforms)
        self.transforms: List[Transform] = transforms

        self._pd_component_list = [t.pd_component for t in self.transforms]
        try:
            self._pd_component = set.union(*self._pd_component_list)
        except (AttributeError, TypeError):
            self._pd_component = None

    def __sub__(self, name: str) -> "Transform":
        """Create a named clone of the transform.

        Example:
            named_t = t - 'rescale image'
        """
        named_copy = copy(self)
        named_copy._pd_name = name
        named_copy._pd_group = True
        named_copy._pd_varname = _notset
        return named_copy

    def __getitem__(self, item: Union[int, slice, str]) -> Transform:
        """Get item

        If int, gets item'th transform in this CompoundTransform.
        If slice, gets sliced transform of same type
        If str, gets first transform with name item

        :param item: Should be of type {int, slice, str}
        """
        if isinstance(item, int):
            return self.transforms[item]
        if isinstance(item, slice):
            return type(self)(self.transforms[item])
        if isinstance(item, str):
            for transform_ in self.transforms:
                if transform_.pd_name == item:
                    return transform_
            raise ValueError(f"{item}: Transform with pd_name '{item}' not found")
        raise TypeError('Unknown type for get item: expected type {int, slice, str}')

    def _pd_evaluable_repr_inner(self, indent=0):
        sub_reprs = [
            x.pd_varname() or x._pd_evaluable_repr(indent + 4)
            for x in self.transforms
        ]
        result = (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )
        if self._pd_group:
            result = 'padl.group' + result
        return result

    def _pd_build_codegraph(self, graph=None, scopemap=None, name=None, scope=None):
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        start = self._pd_codegraph_startnode(name)

        if self._pd_group and 'padl' not in graph:
            emptyscope = symfinder.Scope.empty()
            graph['padl', emptyscope] = var2mod.CodeNode.from_source('import padl', emptyscope)
            scopemap['padl', self._pd_call_info.scope] = emptyscope

        if name is not None:
            assert scope is not None
            graph[name, scope] = start
            scopemap[name, scope] = scope

        for transform in self.transforms:
            varname = transform.pd_varname()
            transform._pd_build_codegraph(graph, scopemap, varname,
                                          self._pd_call_info.scope)
        return graph, scopemap

    def _pd_longrepr(self, formatting=True):
        between = f'\n{make_green(self.display_op, not formatting)}  \n'
        rows = [make_bold(f'{i}: ', not formatting) + t._pd_shortrepr(formatting)
                for i, t in enumerate(self.transforms)]
        return between.join(rows) + '\n'

    def _pd_shortrepr(self, formatting=True):
        def subrepr(transform):
            short = transform._pd_shortrepr(formatting)
            if len(short) < 20:
                return short
            return transform._pd_tinyrepr(formatting)

        result = f' {make_green(self.op, not formatting)} ' \
            .join(subrepr(t) for t in self.transforms)
        if self._pd_group:
            result = f'group({result})'
        return result

    def _pd_tinyrepr(self, formatting=True) -> str:
        rep = f'..{make_green(self.op, not formatting)}..'
        if self.pd_name:
            rep = f'{self.pd_name}: {rep}'
        return f'{make_green("[", not formatting)}{rep}{make_green("]", not formatting)}'

    def _pd_title(self):
        return self.__class__.__name__

    def pd_to(self, device: str):
        """Set the transform's device to *device*

        :param device: device on which to send {'cpu', cuda', 'cuda:N'}
        """
        self._pd_device = device
        for transform_ in self.transforms:
            transform_.pd_to(device)

        self.pd_preprocess._pd_device = device
        self.pd_forward._pd_device = device
        self.pd_postprocess._pd_device = device
        return self

    def _pd_forward_device_check(self):
        """Check all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole CompoundTransform.

        :return: Bool
        """
        return_val = True

        if isinstance(self.pd_forward, type(self)):
            for transform_ in self.pd_forward.transforms:
                if self.pd_device != transform_.pd_device:
                    raise WrongDeviceError(self, transform_)
                return_val = transform_._pd_forward_device_check()
            return return_val

        if self.pd_device != self.pd_forward.pd_device:
            raise WrongDeviceError(self, self.pd_forward)

        return self.pd_forward._pd_forward_device_check()

    @classmethod
    def _flatten_list(cls, transform_list: List[Transform]):
        """Flatten *list_* such that members of *cls* are not nested.

        :param transform_list: List of transforms.
        """
        list_flat = []

        for transform in transform_list:
            if isinstance(transform, cls):
                if transform._pd_group:
                    list_flat.append(transform)
                else:
                    list_flat += transform.transforms
            else:
                list_flat.append(transform)

        return list_flat

    @property
    def _pd_direct_subtransforms(self):
        yield from self.transforms

    def grouped(self):
        """Return a grouped version of *self*. """
        return type(self)(self.transforms, self._pd_call_info, pd_name=self.pd_name, pd_group=True)

    @staticmethod
    def _pd_get_keys(transforms):
        """Get deduplicated keys from list of transforms

        Names are updated as below.
        [None, None, 'a', 'a', 'b', None] -> ['out_0', 'out_1', 'a_0', 'a_1', 'b', 'out_5']

        :param transforms: list of transforms
        :return: list of keys
        """
        names = []
        for ind, transform_ in enumerate(transforms):
            if not transform_.pd_name:
                name = 'out_'+str(ind)
            else:
                name = transform_.pd_name
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
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: name of the Compose transform
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `CompoundTransform`.
    :return: output from series of transforms
    """
    op = '>>'
    display_op = '>>'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

        preprocess_end = 0
        postprocess_start = len(self.transforms)
        set_postprocess = True
        for i, transform_ in enumerate(self.transforms):
            if 'preprocess' in transform_.pd_component:
                preprocess_end = i
            if 'postprocess' in transform_.pd_component and set_postprocess:
                postprocess_start = i
                set_postprocess = False
        for i in range(preprocess_end):
            self._pd_component_list[i] = {'preprocess'}
        for i in range(postprocess_start+1, len(self.transforms)):
            self._pd_component_list[i] = {'postprocess'}

    def _pd_classify_nodetype(self, i, t, t_m1, cw, cw_m1):
        if i > 0 and isinstance(t, Parallel) and len(cw) == len(cw_m1):
            type_ = 'multi_2_multi'

        elif i > 0 and cw == 1 and cw_m1 > 1:
            type_ = 'multi_2_single'

        elif cw == 1 or isinstance(t, Compose):
            type_ = 'single_2_single'

        else:
            type_ = 'single_2_multi'

        return type_

    def _pd_longrepr(self, formatting=True) -> str:  # TODO: make it respect the formatting
        """Create a detailed formatted representation of the transform. For multi-line inputs
        the lines are connected with arrows indicating data flow.
        """
        # pad the components of rows which are shorter than other parts in same column
        rows = [
            [s._pd_parensrepr() for s in t.transforms] if hasattr(t, 'transforms')
            else [t._pd_shortrepr()]
            for t in self.transforms
        ]

        children_widths = [[visible_len(x) for x in row] for row in rows]
        # get maximum widths in "columns"
        children_widths_matrix = np.zeros((len(self.transforms),
                                           max([len(x) for x in children_widths])))
        for i, cw in enumerate(children_widths):
            children_widths_matrix[i, :len(cw)] = cw
        max_widths = np.max(children_widths_matrix, 0)

        for i, r in enumerate(rows):
            for j in range(len(rows[i])):
                if len(rows[i][j]) < max_widths[j]:
                    rows[i][j] += ' ' * (int(max_widths[j]) - len(rows[i][j]))

        for i, r in enumerate(rows):
            if len(r) > 1:
                rows[i] = f' {make_green(self.transforms[i].display_op)} '.join(r)
            else:
                rows[i] = r[0]
        output = []
        # iterate through rows and create arrows depending on numbers of components
        for i, (r, t) in enumerate(zip(rows, self.transforms)):
            widths = [0]
            subarrows = []

            type_ = self._pd_classify_nodetype(i, t, self.transforms[i - 1],
                                               children_widths[i], children_widths[i - 1])

            # if subsequent rows have the same number of "children" transforms
            if type_ == 'multi_2_multi':
                for j, w in enumerate(children_widths[i]):
                    subarrows.append(create_arrow(sum(widths) - j + j * 4, 0, 0, 0))
                    widths.append(int(max_widths[j]))

            # if previous row has multiple outputs and current row just one input
            elif type_ == 'multi_2_single':
                for j, w in enumerate(children_widths[i - 1]):
                    subarrows.append(create_reverse_arrow(
                        0, sum(widths) - j + j * 4,
                        len(children_widths[i - 1]) - j + 1, j + 1
                    ))
                    widths.append(int(max_widths[j]))

            # if previous has single output and current row has single input
            elif type_ == 'single_2_single':
                subarrows.append(create_arrow(0, 0, 0, 0))

            # if previous row has one output and current row has multiple inputs
            else:
                assert type_ == 'single_2_multi'
                for j, w in enumerate(children_widths[i]):
                    if isinstance(t, Rollout):
                        subarrows.append(create_arrow(0, sum(widths) - j + j * 4,
                                                      len(children_widths[i]) - j, j + 1))
                    else:
                        subarrows.append(create_arrow(j, sum(widths) - j + j * 3,
                                                      len(children_widths[i]) - j, j + 1))
                    widths.append(int(max_widths[j]))

            # add signature names to the arrows
            tuple_to_str = lambda x: '(' + ', '.join([str(y) for y in x]) + ')'
            if (isinstance(t, Rollout) or isinstance(t, Parallel)) and not t._pd_name:
                all_params = []
                for tt in t.transforms:
                    all_params.append(list(tt._pd_get_signature().keys()))
                to_combine = [
                    ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + tuple_to_str(params)
                    if len(params) > 1
                    else ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + params[0]
                    for k, params in enumerate(all_params)
                ]
                to_format = combine_multi_line_strings(to_combine)
            else:
                params = [x for x in t._pd_get_signature()]
                to_format = '  ' + tuple_to_str(params) if len(params) > 1 else '  ' + params[0]
            to_format_pad_length = max([len(x.split('\n')) for x in subarrows]) - 1
            to_format = ''.join(['\n' for _ in range(to_format_pad_length)] + [to_format])

            # combine the arrows
            mark = combine_multi_line_strings(subarrows + [to_format])
            mark = '\n'.join(['   ' + x for x in mark.split('\n')])
            output.append(make_green(mark))
            output.append(make_bold(f'{i}: ') + r)
        return '\n'.join(output)

    def __call__(self, args):
        """Call method for Compose

        :param args: Arguments to call with.
        :return: Output from series of transforms.
        """
        for transform_ in self.transforms:
            args = transform_._pd_call_transform(args)
        return args

    @property
    def pd_forward(self) -> Transform:
        if self._pd_forward is None:
            t_list = []
            for transform_, component_set in zip(self.transforms, self._pd_component_list):
                if 'forward' in component_set:
                    if len(component_set) == 1:
                        t_list.append(transform_)
                    else:
                        t_list.append(transform_.pd_forward)

            if len(t_list) == 1:
                self._pd_forward = t_list[0]
            elif t_list:
                self._pd_forward = Compose(t_list, call_info=self._pd_call_info)
            else:
                self._pd_forward = Identity()

        return self._pd_forward

    @property
    def pd_preprocess(self) -> Transform:
        if self._pd_preprocess is None:
            t_list = []
            for transform_, component_set in zip(self.transforms, self._pd_component_list):
                if 'preprocess' in component_set:
                    if len(component_set) == 1:
                        t_list.append(transform_)
                    else:
                        t_list.append(transform_.pd_preprocess)

            if len(t_list) == 1:
                self._pd_preprocess = t_list[0]
            elif t_list:
                self._pd_preprocess = Compose(t_list, call_info=self._pd_call_info)
            else:
                self._pd_preprocess = Identity()

        return self._pd_preprocess

    @property
    def pd_postprocess(self) -> Transform:
        if self._pd_postprocess is None:
            t_list = []
            for transform_, component_set in zip(self.transforms, self._pd_component_list):
                if 'postprocess' in component_set:
                    if len(component_set) == 1:
                        t_list.append(transform_)
                    else:
                        t_list.append(transform_.pd_postprocess)

            if len(t_list) == 1:
                self._pd_postprocess = t_list[0]
            elif t_list:
                self._pd_postprocess = Compose(t_list, call_info=self._pd_call_info)
            else:
                self._pd_postprocess = Identity()
        return self._pd_postprocess


class Rollout(CompoundTransform):
    """Apply a list of transform to same input and get tuple output

    Rollout([t1, t2, ...])(x) := (t1(x), t2(x), ...)

    :param transforms: List of transforms to rollout.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: Name of the transform.
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `CompoundTransform`.
    """
    op = '+'
    display_op = '+'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: str = None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    def __call__(self, args):
        """Call method for Rollout

        :param args: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for transform_ in self.transforms:
            out.append(transform_._pd_call_transform(args))
        if Transform.pd_stage is not None:
            return tuple(out)
        return self._pd_output_format(*out)

    @property
    def pd_preprocess(self) -> Transform:
        if self._pd_preprocess is None:
            t_list = [x.pd_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Rollout(t_list, call_info=self._pd_call_info)
        return self._pd_preprocess

    @property
    def pd_forward(self) -> Transform:
        if self._pd_forward is None:
            t_list = [x.pd_forward for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_forward = Identity()
            elif 'preprocess' in self._pd_component and 'forward' in self._pd_component:
                self._pd_forward = Parallel(t_list, call_info=self._pd_call_info)
            else:
                self._pd_forward = Rollout(t_list, call_info=self._pd_call_info)
        return self._pd_forward

    @property
    def pd_postprocess(self) -> Transform:
        if self._pd_postprocess is None:
            t_list = [x.pd_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_postprocess = Identity()
            elif len(list(self._pd_component)) >= 2 and 'postprocess' in self._pd_component:
                self._pd_postprocess = Parallel(t_list, call_info=self._pd_call_info)
            else:
                self._pd_postprocess = Rollout(t_list, call_info=self._pd_call_info)
        return self._pd_postprocess

    def _pd_longrepr(self, formatting=True) -> str:
        make_green_ = lambda x: make_green(x, not formatting)
        make_bold_ = lambda x: make_bold(x, not formatting)
        between = f'\n{make_green_("│ " + self.display_op)}  \n'
        rows = [make_green_('├─▶ ') + make_bold_(f'{i}: ') + t._pd_shortrepr()
                for i, t in enumerate(self.transforms[:-1])]
        rows.append(make_green_('└─▶ ') + make_bold_(f'{len(self.transforms) - 1}: ')
                    + self.transforms[-1]._pd_shortrepr())
        return between.join(rows) + '\n'


class Parallel(CompoundTransform):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: Name of the transform.
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `CompoundTransform`.
    """
    op = '/'
    display_op = '/'

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    def __call__(self, args):
        """Call method for Parallel

        :param args: Argument to call with.
        :return: Namedtuple of output.
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            out.append(transform_._pd_call_transform(args[ind]))
        if Transform.pd_stage is not None:
            return tuple(out)
        return self._pd_output_format(*out)

    @property
    def pd_preprocess(self) -> Transform:
        if self._pd_preprocess is None:
            t_list = [x.pd_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_preprocess

    @property
    def pd_forward(self) -> Transform:
        if self._pd_forward is None:
            t_list = [x.pd_forward for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_forward = Identity()
            else:
                self._pd_forward = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_forward

    @property
    def pd_postprocess(self) -> Transform:
        if self._pd_postprocess is None:
            t_list = [x.pd_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_postprocess = Identity()
            else:
                self._pd_postprocess = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_postprocess

    def _pd_longrepr(self, formatting=True) -> str:
        if not formatting:
            make_green_ = lambda x: x
            make_bold_ = lambda x: x
        else:
            make_green_ = make_green
            make_bold_ = make_bold
        def pipes(n):
            return "│" * n
        def spaces(n):
            return " " * n
        def horizontal(n):
            return "─" * n
        len_ = len(self.transforms)
        out = ''
        for i, t in enumerate(self.transforms):
            out += (
                make_green_(pipes(len_ - i - 1) + '└' + horizontal(i + 1) + '▶ ') +
                make_bold_(f'{i}: ') + t._pd_shortrepr() + '\n'
            )
            if i < len(self.transforms) - 1:
                out += f'{make_green_(pipes(len_ - i - 1) + spaces(i + 2) + self.display_op)}  \n'
        return out


class BuiltinTransform(AtomicTransform):
    def __init__(self, call):
        super().__init__(call)

    def _pd_build_codegraph(self, graph: Optional[dict] = None,
                            scopemap: Optional[dict] = None,
                            name: Optional[str] = None,
                            scope: Optional[symfinder.Scope] = None) -> Tuple[dict, dict]:
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        if scope is None:
            scope = self._pd_call_info.scope

        if ('padl', scope) not in graph:
            emptyscope = symfinder.Scope.empty()
            graph['padl', emptyscope] = var2mod.CodeNode.from_source('import padl', scope)
            scopemap['padl', scope] = emptyscope

        if name is not None:
            start_source = f'{name or "_pd_dummy"} = {self._pd_evaluable_repr()}'
            graph[name, scope] = \
                var2mod.CodeNode.from_source(start_source, scope)

            scopemap[name, scope] = scope

        return graph, scopemap

    def _pd_longrepr(self, formatting=True):
        return self._pd_call.split('padl.')[-1]


class Identity(BuiltinTransform):
    """Do nothing. Just pass on."""

    def __init__(self):
        super().__init__('padl.Identity()')

    def __call__(self, args):
        return args


class Unbatchify(ClassTransform):
    """Mark start of postprocessing.

    Unbatchify removes batch dimension (inverse of Batchify) and moves the input tensors to 'cpu'.

    :param dim: Batching dimension.
    :param cpu: If *True*, moves output to cpu after unbatchify.
    """

    def __init__(self, dim=0, cpu=True):
        super().__init__(arguments=OrderedDict([('dim', dim), ('cpu', cpu)]))
        self.dim = dim
        self._pd_component = {'postprocess'}
        self.cpu = cpu

    def _move_to_device(self, args):
        if isinstance(args, (tuple, list)):
            return tuple([self._move_to_device(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.to('cpu')
        return args

    def __call__(self, args):
        assert Transform.pd_stage is not None, ('Stage is not set, use infer_apply, eval_apply '
                                                'or train_apply instead of calling the transform '
                                                'directly.')

        if Transform.pd_stage != 'infer':
            return self._move_to_device(args) if self.cpu else args
        if isinstance(args, tuple):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            args = args.squeeze(self.dim)
            return args.to('cpu') if self.cpu else args

        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(ClassTransform):
    """Mark end of preprocessing.

    Batchify adds batch dimension at *dim*. During inference, this unsqueezes tensors and,
    recursively, tuples thereof. Batchify also moves the input tensors to device specified
    for the transform.

    :param dim: Batching dimension.
    """

    def __init__(self, dim=0):
        super().__init__(arguments=OrderedDict([('dim', dim)]))
        self.dim = dim
        self._pd_component = {'preprocess'}

    def __call__(self, args):
        assert Transform.pd_stage is not None, ('Stage is not set, use infer_apply, eval_apply '
                                                'or train_apply instead of calling the transform '
                                                'directly.')

        if Transform.pd_stage != 'infer':
            return args
        if isinstance(args, (tuple, list)):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim)
        if isinstance(args, (float, int)):
            return torch.tensor([args])
        raise TypeError('only tensors and tuples of tensors recursively supported...')


def save(transform: Transform, path: Union[Path, str], force_overwrite: bool = False):
    """Save the transform to a folder at *path*.

    The folder's name should end with '.padl'. If no extension is given, it will be added
    automatically.

    If the folder exist, call with *force_overwrite* = `True` to overwrite. Otherwise, this
    will raise a FileExistsError.
    """
    transform.pd_save(path, force_overwrite)


def load(path):
    """Load a transform (as saved with padl.save) from *path*. """
    path = Path(path)
    with open(path / 'transform.py') as f:
        source = f.read()
    module = types.ModuleType('lfload')
    module.__dict__.update({
        '_pd_source': source,
        '_pd_module': module,
        '__file__': str(path / 'transform.py')
    })
    code = compile(source, path/'transform.py', 'exec')
    exec(code, module.__dict__)
    transform = module._pd_main
    for i, subtrans in enumerate(transform._pd_all_transforms()):
        subtrans.pd_post_load(path, i)
    return transform


def group(transform: Union[Rollout, Parallel]):
    """Group transforms. This prevents them from being flattened when used

    Example:
    When writing a Rollout as `(a + (b + c))`, this is automatically flattened to `(a + b + c)`
    - i.e. the resulting Rollout transform expects a 3-tuple whose inputs are passed to `a`, `b`,
    `c` respectively. To prevent that, do (a + group(b + c)). The resulting Rollout will expect a
    2-tuple whose first item will be passed to `a` and whose second item will be passed to `b + c`.
    """
    return transform.grouped()
