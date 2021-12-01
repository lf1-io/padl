"""The Transform class and its fundamental children.

Transforms should be created using the `padl.transform` wrap-function.
"""
import re
from copy import copy
from collections import Counter, namedtuple, OrderedDict
import contextlib
import inspect
from itertools import chain
from pathlib import Path
from os import remove
from shutil import rmtree
import textwrap
import traceback
from tempfile import TemporaryDirectory
import types
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Set, Tuple, Union
from warnings import warn
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from padl.dumptools import var2mod, symfinder, inspector
from padl.dumptools.symfinder import ScopedName
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

_pd_trace = []


def _unpack_batch(args):
    """Convert an input in batch-form into a tuple of datapoints.

    E.g:

        ([1, 4], ([2, 5], [3, 6])) -> [(1, (2, 3)), (4, (5, 6))]

    :param args: arguments to be unbatched
    """
    out = []
    itr = 0
    while True:
        try:
            temp = _batch_get(args, itr)
            out.append(temp)
            itr += 1
        except IndexError:
            return out


def _batch_get(args, i):
    """Get the *i*th element of a tensor
    or
    get a tuple of the *i*th elements of a tuple (or list) of tensors

    >>> t1 = torch.Tensor([1,2,3])
    >>> t2 = torch.Tensor([4,5,6])
    >>> _batch_get(t1, 1)
    tensor(2.)
    >>> _batch_get((t1, t2), 1)
    (tensor(2.), tensor(5.))

    :param args: arguments
    :param i: index in batch
    """
    if isinstance(args, torch.Tensor):
        return args[i]
    if isinstance(args, list):
        return [_batch_get(args[j], i) for j in range(len(args))]
    if isinstance(args, tuple):
        return tuple([_batch_get(args[j], i) for j in range(len(args))])
    if isinstance(args, dict):
        return {k: _batch_get(args[k], i) for k in args}
    raise TypeError


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
    if isinstance(args, tuple):
        return tuple([_move_to_device(x, device) for x in args])
    if isinstance(args, list):
        return [_move_to_device(x, device) for x in args]
    if isinstance(args, torch.Tensor):
        return args.to(device)
    return args


Mode = Literal['infer', 'eval', 'train']
Stage = Literal['preprocess', 'forward', 'postprocess']


class Transform:
    """Transform base class.

    :param call_info: A `CallInfo` object containing information about the how the transform was
    created (needed for saving).
    :param pd_name: name of the transform
    """
    pd_mode = None

    def __init__(self, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        if call_info is None:
            call_info = inspector.CallInfo()
        self._pd_call_info = call_info
        self._pd_varname = _notset
        self._pd_name = pd_name
        self._pd_device = 'cpu'
        self._pd_layers = None
        self._pd_traceback = traceback.extract_stack()
        self._pd_stages = None

    @staticmethod
    def _pd_merge_components(components):
        """Merge components recusively such that lists consisting of all the same integers are
        merged to that integer.

        >>> Transform._pd_merge_components(0)
        0
        >>> Transform._pd_merge_components([1, 1, 1])
        1
        >>> Transform._pd_merge_components([1, 2, [1, 1]])
        [1, 2, 1]
        """
        if isinstance(components, int):
            return components
        if all(isinstance(x, int) for x in components) and len(set(components)) == 1:
            return components[0]
        res = [Transform._pd_merge_components(x) for x in components]
        if res != components:
            return Transform._pd_merge_components(res)
        return res

    def _pd_get_stages(self):
        if self._pd_stages is None:
            _, splits, has_batchify = self._pd_splits()
            if has_batchify:
                preprocess, forward, postprocess = splits
            else:
                preprocess, forward, postprocess = builtin_identity, splits[0], splits[2]
            self._pd_stages = preprocess, forward, postprocess

        return self._pd_stages

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple['Transform',
                                                            'Transform',
                                                            'Transform'],
                                                      bool]:
        """ Split the transform into "pre-batchified", "batchified" and "postprocessing" splits.

        *input_components* contains information about the "split" the input is in and potentially
        how many "pipes" of input there are. It is either an int or a (potentially nested)
        list of ints. A list of ints indicates there are multiple "pipes".
        The ints have the following meaning:
            - 0 means "not batchified"
            - 1 means "batchified"
            - 2 means "unbatchified" a.k.a. post-process
        If it's a (nested) list, the structure represents the input structure for the transform,
        whereas the entries represent the "split" of parts of the input.

        For example, if a transform expects a tuple of inputs, *input_components* could be
        (0, 1), meaning that the first input item is not batchified whereas the second is.

        The method returns a tuple (*output_components*, *splits*).

        - *output_components* is the "splits" information of the output, it has the same format as
        the *input_components*.

        - *splits* is a 3-tuple of splits, the entries are:
            - the "pre-batchified" part of the transform
            - the "batchified" part of the transform
            - the "postprocess" part of the transform
        """
        component = Transform._pd_merge_components(input_components)
        assert isinstance(component, int), ('A normal Tranform cannot process input from multiple '
                                            'stages.')
        return (
            # a normal transform doesn't change the components
            component,
            # for the component the transform is in, return the transform, else Identity
            tuple(self if i == component else builtin_identity for i in range(3)),
            False
        )

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
        try:
            return self.pre_save(path, i)
        except AttributeError:
            pass

    def pd_post_load(self, path: Path, i: int):
        """Method that is called on each transform after loading.

        This normally does nothing. Override to implement custom serialization.

        :param path: The load path.
        :param i: Unique transform index, can be used to construct filenames.
        """
        try:
            return self.post_load(path, i)
        except AttributeError:
            pass

    def pd_zip_save(self, path: Union[Path, str], force_overwrite: bool = False):
        """Save the transform to a zip-file at *path*.

        The file's name should end with '.padl'. If no extension is given, it will be added
        automatically.

        If the file exists, call with *force_overwrite* = `True` to overwrite. Otherwise, this
        will raise a FileExistsError.
        """
        path = Path(path)
        if path.suffix == '':
            path = path.parent / (path.name + '.padl')

        if path.exists():
            if not force_overwrite:
                raise FileExistsError(f'{path} exists, call with *force_overwrite* to overwrite.')
            try:
                rmtree(path)
            except NotADirectoryError:
                remove(path)

        with TemporaryDirectory('.padl') as dirname:
            self.pd_save(dirname, True)
            with ZipFile(path, 'w') as zipf:
                for file in Path(dirname).glob('*'):
                    if file.is_file():
                        zipf.write(file, file.name)

    def pd_save(self, path: Union[Path, str], force_overwrite: bool = False):
        """Save the transform to a folder at *path*.

        The folder's name should end with '.padl'. If no extension is given, it will be added
        automatically.

        If the folder exists, call with *force_overwrite* = `True` to overwrite. Otherwise, this
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
        return var2mod.CodeNode.from_source(start_source, self._pd_call_info.scope)

    @property
    def _pd_closurevars(self) -> Tuple[dict, dict]:
        """Return the closurevars (globals and nonlocals) the transform depends on. """
        return {}, {}

    def _pd_build_codegraph(self, graph: Optional[dict] = None,
                            scopemap: Optional[dict] = None,
                            name: Optional[str] = None,
                            scope: Optional[symfinder.Scope] = None) -> Tuple[dict, dict]:
        """Build a codegraph defining the transform.

        A codegraph's nodes are :class:`CodeNode` instances which contain a scoped name, a piece of
        code defining the name and a set of dependencies (other scoped names). The dependencies
        can be understood as the edges in the graph.

        A transform's codegraph starts with a "start-node", which is a :class:`CodeNode`
        representing an assignment of the transform's evaluable representation to a variable
        called *name*.

        From there, iteratively, all :class:`CodeNode`s representing the existing dependencies are
        searched for and added to the graph.

        Example:

        Given the following code ...::

            from padl import transform

            a = 100

            @transform
            def f(x):
                return x + a

        ... ``f._pd_build_codegraph(name='mytransform')`` would first create the start-node::

            "mytransform": CodeNode(source='mytransform = f',
                                    globals_={('f', 0)})

        By iterating though the dependencies ("globals_"), the code-node of 'f' would be added::

            "f": CodeNode(source='@transform\ndef f(x):\n [...]',
                          globals_={('transform', 0), ('a', 0)})

        This adds two new dependencies for which code-nodes are added::

            "transform": CodeNode(source='from padl import transform', globals_={})
            "a": CodeNode(source='a = 100', globals_={})

        This leaves no more dependencies to be found. These four nodes are ``f``'s codegraph.
        The codegraph can be used to compile a python module defining ``f``.

        :param graph: A codegraph to extend. If *None* a new codegraph will be created.
        :param scopemap: A dict mapping scoped names to the scopes they were created in.
        :param name: The name to give the transform.
        :param scope: The scope for the start-node. Default is to use the scope of the transform.
        :return: Updated graph and scopemap.
        """
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        # the default is to use the scope of the transform
        if scope is None:
            scope = self._pd_call_info.scope

        # build the start node ->
        # if the *name* is the same as the call, we don't need to assign to the name
        # this can be the case for function transforms
        if getattr(self, '_pd_call', None) == name:
            new_name = None
        else:
            new_name = name

        start = self._pd_codegraph_startnode(new_name)
        # <-

        # if this has closurevars, get them (if there are transforms in the closure, we want to
        # allow them to build their codegraph themselves, see below)
        globals_dict, nonlocals_dict = self._pd_closurevars
        all_vars_dict = {**globals_dict, **nonlocals_dict}

        # find dependencies
        todo = {*start.globals_}
        while todo and (next_name := todo.pop()):
            # we know this already - go on
            if next_name in scopemap:
                continue

            # ignoring this (it comes from the serializer)
            if next_name.name.startswith('PADL_VALUE'):
                continue

            # see if the object itself knows how to generate its codegraph
            try:
                if next_name.scope.is_global():
                    next_obj = globals_dict[next_name.name]
                else:
                    next_obj = all_vars_dict[next_name.name]
                # pylint: disable=protected-access
                next_obj._pd_build_codegraph(graph, scopemap, next_name.name)
            except (KeyError, AttributeError):
                pass
            else:
                continue

            # find how next_ came into being
            (source, node), scope_of_next_var = symfinder.find_in_scope(next_name)

            # store a mapping from *next_name* to it's defining scope
            scopemap[next_name] = scope_of_next_var

            # find dependencies
            dependencies = var2mod.find_globals(node)
            # fix the scope of *next_name* (from where it was a dependency to where it was defined)
            next_name = ScopedName(next_name.name, scope_of_next_var, next_name.n)
            dependencies = var2mod.increment_same_name_var(dependencies, next_name)

            graph[next_name] = var2mod.CodeNode(source=source,
                                                globals_=dependencies,
                                                ast_node=node)
            todo.update(dependencies)
        # finding dependencies done

        # if *new_name* is not ``None``, add the start node (i.e. the node assigning the transform
        # to *new_name*) to the codegraph
        if new_name is not None:
            assert scope is not None
            graph[ScopedName(new_name, scope, 0)] = start

        if name is not None:
            scopemap[ScopedName(name, scope, 0)] = self._pd_call_info.scope

        return graph, scopemap

    def _pd_process_traceback(self):
        """ Find where the Transform was defined (file, lineno, file) given the traceback. """
        a_tb = None
        for a_tb in self._pd_traceback[::-1]:
            if 'padl/transforms' in a_tb[0] or 'padl/util_transforms' in a_tb[0]:
                continue
            break
        return f'{a_tb.filename} in {a_tb.name}\n----> {a_tb.lineno} {a_tb.line}'

    def _pd_trace_error(self, position: int, arg):
        """ Add some error description to `pd_trace`. """
        try:
            breakpoint()
            str_ = self._pd_longrepr()
            _pd_trace.append((str_, self._pd_process_traceback(), arg, self))
        except Exception:
            warn('Error tracing failed')

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

    def pd_varname(self, module=None) -> Optional[str]:
        """The name of the variable name the transform was last assigned to.

        Example:

        >>> from padl import transform
        >>> foo = transform(lambda x: x + 1)
        >>> foo.pd_varname()
        'foo'

        :param module: Module to search
        :return: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        if self._pd_varname is _notset or module is not None:
            if module is None:
                module = inspector.caller_module()
            self._pd_varname = self._pd_find_varname(module.__dict__)
        return self._pd_varname

    def pd_forward_device_check(self) -> bool:
        """Check if all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole Pipeline.
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

    def _pd_call_transform(self, arg):
        try:
            if self._pd_unpack_argument(arg):
                return self(*arg)
            return self(arg)
        except Exception as err:
            self._pd_trace_error(0, arg)
            raise err

    def pd_call_transform(self, arg, mode: Optional[Mode] = None):
        """Call the transform, with possibility to pass multiple arguments.

        :param arg: argument to call the transform with
        :param mode: The mode ("infer", "eval", "train") to perform the call with.
        :return: Whatever the transform returns.
        """

        if mode in ('eval', 'infer'):
            torch_context = torch.no_grad()
        else:
            torch_context = contextlib.suppress()

        with self.pd_set_mode(mode), torch_context:
            return self._pd_call_transform(arg)

    def _pd_get_signature(self):
        """Get the signature of the transform. """
        return inspect.signature(self).parameters

    def _pd_get_output_format(self):
        return None

    def _pd_itercall(self, args, mode: Mode, loader_kwargs: Optional[dict] = None,
                     verbose: bool = False, flatten: bool = False) -> Iterator:
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param mode: Mode to call in ("eval", "train" or "infer")
        :param loader_kwargs: Data loader keyword arguments.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.

        :return: A generator that allows iterating over the output.
        """
        assert mode in ('eval', 'train'), '_pd_itercall can only be used with mode eval or train'

        global _pd_trace
        _pd_trace = []

        self.pd_forward_device_check()

        preprocess = self.pd_preprocess
        forward = self.pd_forward
        post = self.pd_postprocess

        use_preprocess = not isinstance(preprocess, Identity)
        use_forward = not isinstance(forward, Identity)
        use_post = not isinstance(post, Identity)

        if use_preprocess:
            loader = self.pd_get_loader(args, preprocess, mode, **loader_kwargs)
        else:
            loader = args

        pbar = None
        if verbose:
            if use_post or flatten:
                pbar = tqdm(total=len(args))
            else:
                loader = tqdm(loader, total=len(loader))

        for batch in loader:
            batch = _move_to_device(batch, self.pd_device)

            output = batch
            if use_forward:
                output = forward.pd_call_transform(batch, mode)

            if use_post or flatten:
                if verbose:
                    pbar.update()

                output = _unpack_batch(output)
                if use_post:
                    output = [post.pd_call_transform(x, mode) for x in output]
                for out in output:
                    output_format = self._pd_get_output_format()
                    if output_format is not None:
                        yield output_format(*out)
                    else:
                        yield out
            else:
                output_format = self._pd_get_output_format()
                if output_format is not None:
                    yield output_format(*output)
                else:
                    yield output

    @property
    def pd_device(self) -> str:
        """Return the device ("cpu" / "cuda") the transform is on."""
        return self._pd_device

    @property
    def pd_preprocess(self) -> "Transform":
        """The preprocessing part of the transform. The device must be propagated from self."""
        pre = self._pd_get_stages()[0]
        pre.pd_to(self.pd_device)
        return pre

    @property
    def pd_forward(self) -> "Transform":
        """The forward part of the transform (that what's typically done on the GPU).
        The device must be propagated from self."""
        forward = self._pd_get_stages()[1]
        forward.pd_to(self.pd_device)
        return forward

    @property
    def pd_postprocess(self) -> "Transform":
        """The postprocessing part of the transform. The device must be propagated from self."""
        post = self._pd_get_stages()[2]
        post.pd_to(self.pd_device)
        return post

    def pd_to(self, device: str) -> "Transform":
        """Set the transform's device to *device*.

        :param device: Device to set the transform to {'cpu', 'cuda', 'cuda:N'}.
        """
        self._pd_device = device
        for layer in self.pd_layers:
            layer.to(device)
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
    def pd_set_mode(self, mode: Optional[str] = None):
        """Set of mode of Transform

        :param mode: mode ('train', 'eval', 'infer')
        """
        assert mode in ('train', 'eval', 'infer', None)

        if mode is None:
            yield
            return

        layers = self.pd_layers
        training_before = [layer.training for layer in layers]
        try:
            for layer in layers:
                if mode == 'train':
                    layer.train()
                else:
                    layer.eval()
            Transform.pd_mode = mode
            yield
        finally:
            for i, training in enumerate(training_before):
                layer = layers[i]
                if training:
                    layer.train()
                else:
                    layer.eval()
            Transform.pd_mode = None

    @staticmethod
    def pd_get_loader(args, preprocess, mode, **kwargs) -> DataLoader:
        """Get a pytorch data loader.

        :param args: A sequence of datapoints.
        :param preprocess: preprocessing step
        :param mode: mode
        :param kwargs: Keyword arguments passed to the data loader (see the pytorch
            `DataLoader` documentation for details).
        """
        sequence = _ItemGetter(
            args,
            lambda *args: preprocess.pd_call_transform(*args, mode),
        )

        return DataLoader(
            sequence,
            worker_init_fn=lambda _: np.random.seed(),
            **kwargs
        )

    def infer_apply(self, inputs):
        """Call transform within the infer context.

        This expects a single argument and returns a single output.

        :param inputs: The input.
        """
        self.pd_forward_device_check()
        inputs = self.pd_preprocess.pd_call_transform(inputs, mode='infer')
        inputs = _move_to_device(inputs, self.pd_device)
        inputs = self.pd_forward.pd_call_transform(inputs, mode='infer')
        inputs = self.pd_postprocess.pd_call_transform(inputs, mode='infer')
        output_format = self._pd_get_output_format()
        if output_format is not None:
            return output_format(*inputs)
        else:
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
    other transforms - in contrast to :class:`Pipeline`).

    Examples of :class:`AtomicTransform` s are :class:`ClassTransform` and
    :class:`FunctionTransform`.

    :param call: The transform's call string.
    :param call_info: A :class:`CallInfo` object containing information about the how the
        transform was created (needed for saving).
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
        """Return the closurevars (globals and nonlocals) the transform depends on.

        Closurevars are variables that are used inside a transform but weren't define there.

        Example:

        In this case...

            z = 100
            def make_transform():
                b = 1
                @transform
                def f(x):
                    a = 10
                    return a + x + z + b

        ... "f" has a global closurevar "z" (defined in the global scope) and nonlocal closurevar
        "b" (defined in the scope surrounding "f", but not the global scope).
        """
        try:
            closurevars = inspect.getclosurevars(self.function)
        except TypeError as exc:
            warn(f'Could not get closurevars ({exc}). This is usually fine as closurevars are only '
                 'needed for user defined transforms.',
                 RuntimeWarning)
            return {}, {}
        return (
            {k: v for k, v in closurevars.globals.items() if v is not self},
            {k: v for k, v in closurevars.nonlocals.items() if v is not self}
        )

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
        (body_msg, _), _ = symfinder.find_in_scope(ScopedName(self.__class__.__name__,
                                                              self._pd_call_info.scope))
        try:
            return 'class ' + body_msg.split('class ', 1)[1]
        except IndexError:
            return body_msg

    def _split_call(self):
        return symfinder.split_call(self._pd_call)

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

    def pre_save(self, path: Path, i: int):
        """Dump the model's parameters to a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        path = Path(path)
        checkpoint_path = path / f'{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def post_load(self, path, i):
        """Load the model's parameters form a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        path = Path(path)
        checkpoint_path = path / f'{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))

    def _pd_longrepr(self) -> str:
        return torch.nn.Module.__repr__(self)


class Map(Transform):
    """Apply one transform to each element of a list.

    >>> from padl import identity
    >>> t = identity
    >>> x1, x2, x3 = 1, 2, 3
    >>> Map(t)([x1, x2, x3]) == (t(x1), t(x2), t(x3))
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

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple[Transform,
                                                            Transform,
                                                            Transform],
                                                      bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        The *splits* of a map are:
            - the map of the subtransform's preprocess
            - the map of the subtransform's forward
            - the map of the subtransform's postprocess

        The *output_components* of a map are the mapped input components.
        """
        # if *input_components* is an integer rather than a list ...
        if isinstance(input_components, int):
            # ... get output_components and splits from the contained transform
            output_components, splits, has_batchify = self.transform._pd_splits(input_components)
            return (
                # output_components is whatever the sub-transform does to it
                output_components,
                # the splits are the splits of the sub-transform, but mapped
                tuple(Map(split) if not isinstance(split, Identity) else builtin_identity
                      for split in splits),
                has_batchify
            )
        assert isinstance(input_components, list)

        # if it's a list, this means the input is structured and can potentially be in
        # different states (fresh, batchified, unbatchified)
        splits = ([], [], [])
        output_components = []
        has_batchify = False
        for input_component in input_components:
            # for each input, we compute the *output_components* and the *splits* ...
            sub_output_components, sub_splits, sub_has_batchify = \
                self.transform._pd_splits(input_component)
            has_batchify = has_batchify or sub_has_batchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # .. and combine them as a Parallel
        return (
            output_components,
            tuple(Parallel(s) if s else builtin_identity for s in splits),
            has_batchify
        )

    def __call__(self, args: Iterable):
        """
        :param args: Args list to call transforms with
        """
        return tuple([self.transform.pd_call_transform(arg) for arg in args])

    def _pd_longrepr(self, formatting=True) -> str:
        return '~ ' + self.transform._pd_shortrepr(formatting)

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
            graph[ScopedName(name, scope, 0)] = start
            scopemap[ScopedName(name, scope, 0)] = scope

        varname = self.transform.pd_varname(self._pd_call_info.module)
        self.transform._pd_build_codegraph(graph, scopemap, varname,  self._pd_call_info.scope)
        return graph, scopemap


class Pipeline(Transform):
    """Abstract base class for Pipeline

    :param transforms: list of transforms
    :param call_info: A `CallInfo` object containing information about the how the transform was
    created (needed for saving).
    :param pd_name: name of Pipeline
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `Pipeline`.
    """
    op = NotImplemented
    display_op = NotImplemented

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(call_info, pd_name)

        self._pd_group = True if pd_name is not None else pd_group

        transforms = self._flatten_list(transforms)
        self.transforms: List[Transform] = transforms

    def _pd_get_output_format(self):
        last_transform = self.transforms[-1]
        if hasattr(last_transform, '_pd_output_format'):
            return last_transform._pd_output_format
        return None

    def _pd_get_output_format(self):
        last_transform = self.transforms[-1]
        if hasattr(last_transform, '_pd_output_format'):
            return last_transform._pd_output_format
        return None

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

        If int, gets item'th transform in this Pipeline.
        If slice, gets sliced transform of same type
        If str, gets first transform with name item

        :param item: Should be of type {int, slice, str}
        """
        if isinstance(item, int):
            return self.transforms[item]
        if isinstance(item, slice):
            transform_ = type(self)(self.transforms[item])
            transform_.pd_to(self.pd_device)
            return transform_
        if isinstance(item, str):
            for transform_ in self.transforms:
                if transform_.pd_name == item:
                    return transform_
            raise ValueError(f"{item}: Transform with pd_name '{item}' not found")
        raise TypeError('Unknown type for get item: expected type {int, slice, str}')

    def __len__(self):
        return len(self.transforms)

    def _pd_call_transform(self, arg):
        if self._pd_unpack_argument(arg):
            return self(*arg)
        return self(arg)

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
        """Build a codegraph defining the transform.

        See :meth:`Transform._pd_build_codegraph` for an explanation of what a code-graph is.

        The codegraph of a :class:`Pipeline` is the union of the codegraphs of the
        contained transforms plus the node defining the transform itself.
        """
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        start = self._pd_codegraph_startnode(name)

        if self._pd_group and 'padl' not in graph:
            emptyscope = symfinder.Scope.empty()
            graph[ScopedName('padl', emptyscope, 0)] = var2mod.CodeNode.from_source('import padl',
                                                                                    emptyscope)
            scopemap[ScopedName('padl', self._pd_call_info.scope, 0)] = emptyscope

        # if a name is given, add the start-node to the codegraph
        if name is not None:
            assert scope is not None
            graph[ScopedName(name, scope, 0)] = start
            scopemap[ScopedName(name, scope, 0)] = scope

        # iterate over sub-transforms and update the codegraph with their codegraphs
        for transform in self.transforms:
            varname = transform.pd_varname(self._pd_call_info.module)
            # pylint: disable=protected-access
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
        return self

    def pd_forward_device_check(self):
        """Check all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole Pipeline.

        :return: Bool
        """
        return_val = True

        if isinstance(self.pd_forward, type(self)):
            for transform_ in self.pd_forward.transforms:
                if self.pd_device != transform_.pd_device:
                    raise WrongDeviceError(self, transform_)
                return_val = transform_.pd_forward_device_check()
            return return_val

        if self.pd_device != self.pd_forward.pd_device:
            raise WrongDeviceError(self, self.pd_forward)

        return self.pd_forward.pd_forward_device_check()

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
        transform_ = type(self)(self.transforms, self._pd_call_info, pd_name=self.pd_name,
                                pd_group=True)
        transform_.pd_to(self.pd_device)
        return transform_

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


class Compose(Pipeline):
    """Apply series of transforms on input.

    Compose([t1, t2, t3])(x) = t3(t1(t2(x)))

    :param transforms: List of transforms to compose.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: name of the Compose transform
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `Pipeline`.
    :return: output from series of transforms
    """
    op = '>>'
    display_op = '>>'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

    def _pd_splits(self, input_components=0, has_batchify=False) -> Tuple[Union[int, List],
                                                                          Tuple[Transform,
                                                                                Transform,
                                                                                Transform],
                                                                          bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        The composition of `transforms` splits into
            - the composition of each sub-transform's preprocess
            - the composition of each sub-transform's forward
            - the composition of each sub-transform's postprocess

        The *output_components* are computed by passing the *input_component* through all
        sub-transforms.
        """

        splits = ([], [], [])

        output_components = input_components
        has_batchify = False
        # for each sub-transform ...
        for transform_ in self.transforms:
            # ... see what comes out ...
            output_components, sub_splits, sub_has_batchify = \
                transform_._pd_splits(output_components)

            has_batchify = has_batchify or sub_has_batchify

            # ... and combine
            # the preprocess split is the composition of the
            # preprocess splits of all subtransforms
            # (same for forward and postprocess)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # .. some cleanup - remove identities ..
        cleaned_splits = tuple(
            [s for s in split if not isinstance(s, Identity)]
            for split in splits
        )

        final_splits = []
        for split in cleaned_splits:
            if len(split) > 1:  # combine sub_splits
                final_splits.append(Compose(split))
            elif len(split) == 1:  # if it's just one, no need to combine
                final_splits.append(split[0])
            else:  # if it's empty: identity
                final_splits.append(builtin_identity)

        return output_components, final_splits, has_batchify

    @staticmethod
    def _pd_classify_nodetype(i, t, t_m1, cw, cw_m1):
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
                params = t._pd_get_signature()
                to_format = '  ' + tuple_to_str(params) if len(params) > 1 else '  ' + \
                    list(params)[0]
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
        _in_args = args
        for i, transform_ in enumerate(self.transforms):
            try:
                args = transform_.pd_call_transform(args)
            except Exception as err:
                self._pd_trace_error(i, _in_args)
                raise err
        return args


class Rollout(Pipeline):
    """Apply a list of transform to same input and get tuple output

    Rollout([t1, t2, ...])(x) := (t1(x), t2(x), ...)

    :param transforms: List of transforms to rollout.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: Name of the transform.
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `Pipeline`.
    """
    op = '+'
    display_op = '+'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: str = None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple[Transform,
                                                            Transform,
                                                            Transform],
                                                      bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        A rollout splits into:
            - the rollout of its sub-transform' first non-Identity split
            - the parallel of its sub-transform' remaining splits

        To see why the first non-Identity split is a rollout whereas the remaining splits are
        parallel, note that the first non-Identity split splits the pipeline and it remains
        split for the rest:

            Case 1:     Case 2:     Case 3:
            pre + pre   Identity    Identity
            for / for   for + for   Identity
            pos / pos   pos / pos   pos + pos

        The *output_components* are the list of output components of the sub-transforms.
        """
        splits = ([], [], [])
        output_components = []
        has_batchify = False
        for transform_ in self.transforms:
            sub_output_components, sub_splits, sub_has_batchify = transform_._pd_splits(input_components)
            has_batchify = has_batchify or sub_has_batchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # only replace with builtin_identity if all Identity to preserve number of pipes

        merged_components = self._pd_merge_components(input_components)
        if not isinstance(merged_components, int):
            merged_components = 0

        cleaned_splits = []
        for i, split in enumerate(splits):
            if all(isinstance(s, Identity) for s in split):
                if i != merged_components:
                    cleaned_splits.append(builtin_identity)
                else:
                    cleaned_splits.append(split)
            else:
                cleaned_splits.append(split)

        first_non_identity = \
            [i for i, s in enumerate(cleaned_splits) if not isinstance(s, Identity)]
        if len(first_non_identity) == 0:
            # Catches scenario where all splits are Identities
            first_non_identity = 0
        else:
            first_non_identity = first_non_identity[0]

        final_splits = []
        for i, s in enumerate(cleaned_splits):
            if isinstance(s, list):
                if i == first_non_identity:
                    final_splits.append(Rollout(s))
                else:
                    final_splits.append(Parallel(s))
            else:
                final_splits.append(s)
        final_splits = tuple(final_splits)

        return output_components, final_splits, has_batchify

    def __call__(self, args):
        """Call method for Rollout

        :param args: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for i, transform_ in enumerate(self.transforms):
            try:
                out.append(transform_.pd_call_transform(args))
            except Exception as err:
                self._pd_trace_error(i, args)
                raise err
        if Transform.pd_mode is not None:
            return tuple(out)
        return self._pd_output_format(*out)

    def _pd_longrepr(self, formatting=True) -> str:
        make_green_ = lambda x: make_green(x, not formatting)
        make_bold_ = lambda x: make_bold(x, not formatting)
        between = f'\n{make_green_("│ " + self.display_op)}  \n'
        rows = [make_green_('├─▶ ') + make_bold_(f'{i}: ') + t._pd_shortrepr()
                for i, t in enumerate(self.transforms[:-1])]
        rows.append(make_green_('└─▶ ') + make_bold_(f'{len(self.transforms) - 1}: ')
                    + self.transforms[-1]._pd_shortrepr())
        return between.join(rows) + '\n'


class Parallel(Pipeline):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: Name of the transform.
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        `Pipeline`.
    """
    op = '/'
    display_op = '/'

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple[Transform, Transform, Transform],
                                                      bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        A parallel splits into:
            - the parallel of its sub-transforms' preprocess
            - the parallel of its sub-transforms' forward
            - the parallel of its sub-transforms' postprocess

        The *output_components* are the list of output components of the sub-transforms.
        """
        splits = ([], [], [])
        # we need one component info per sub-transform - if it's not a list that means
        # all are the same - we make it a list
        input_components_ = input_components
        if not isinstance(input_components_, list):
            input_components_ = [input_components for _ in range(len(self.transforms))]

        # go through the sub-transforms ...
        output_components = []
        has_batchify = False
        for transform_, input_component in zip(self.transforms, input_components_):
            # and compute the sub-splits
            sub_output_components, sub_splits, sub_has_batchify = \
                transform_._pd_splits(input_component)
            has_batchify = has_batchify or sub_has_batchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # only replace with builtin_identity if all Identity to preserve number of pipes
        cleaned_splits = tuple(
            builtin_identity if all(isinstance(s, Identity) for s in split) else split
            for split in splits
        )

        final_splits = tuple(Parallel(s) if isinstance(s, list) else s for s in cleaned_splits)
        return output_components, final_splits, has_batchify

    def __call__(self, args):
        """Call method for Parallel

        :param args: Argument to call with.
        :return: Namedtuple of output.
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            try:
                out.append(transform_.pd_call_transform(args[ind]))
            except Exception as err:
                self._pd_trace_error(ind, args)
                raise err
        if Transform.pd_mode is not None:
            return tuple(out)
        return self._pd_output_format(*out)

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
        caller_frameinfo = inspector.non_init_caller_frameinfo()
        call_info = inspector.CallInfo(caller_frameinfo)
        super().__init__(call, call_info=call_info)

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

        # if padl is not in the scope, add it
        if ScopedName('padl', scope, 0) not in graph:
            emptyscope = symfinder.Scope.empty()
            graph[ScopedName('padl', emptyscope, 0)] = var2mod.CodeNode.from_source('import padl',
                                                                                    scope)
            scopemap[ScopedName('padl', scope, 0)] = emptyscope

        if name is not None:
            start_source = f'{name or "_pd_dummy"} = {self._pd_evaluable_repr()}'
            graph[ScopedName(name, scope, 0)] = \
                var2mod.CodeNode.from_source(start_source, scope)

            scopemap[ScopedName(name, scope, 0)] = scope

        return graph, scopemap

    def _pd_longrepr(self, formatting=True):
        return self._pd_call.split('padl.')[-1]


class Identity(BuiltinTransform):
    """Do nothing. Just pass on."""

    def __init__(self):
        super().__init__('padl.Identity()')

    def __call__(self, args):
        return args


builtin_identity = Identity()


class Unbatchify(ClassTransform):
    """Mark start of postprocessing.

    Unbatchify removes batch dimension (inverse of Batchify) and moves the input tensors to 'cpu'.

    :param dim: Batching dimension.
    :param cpu: If *True*, moves output to cpu after unbatchify.
    """

    def __init__(self, dim=0, cpu=True):
        super().__init__(arguments=OrderedDict([('dim', dim), ('cpu', cpu)]))
        self.dim = dim
        self.cpu = cpu

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple[Transform, Transform, Transform],
                                                      bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        Unbatchify has empty preprocess and forward splits and puts the component-number
        to 2 ("un-batchified").
        """
        # put the output component to 2 ("un-batchified")
        return 2, (builtin_identity, builtin_identity, self), False

    def _move_to_device(self, args):
        if isinstance(args, tuple):
            return tuple([self._move_to_device(x) for x in args])
        if isinstance(args, list):
            return [self._move_to_device(x) for x in args]
        if isinstance(args, torch.Tensor):
            return args.to('cpu')
        return args

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode != 'infer':
            return self._move_to_device(args) if self.cpu else args
        if isinstance(args, tuple):
            return tuple([self(x) for x in args])
        if isinstance(args, list):
            return [self(x) for x in args]
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

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple[Transform, Transform, Transform],
                                                      bool]:
        """See the docstring of :meth:`Transform._pd_splits` for more details.

        Batchify has empty pre-batchified and postprocess splits and puts the component-number
        to 1 ("batchified").
        """
        # ensure that all inputs are "fresh"
        assert self._pd_merge_components(input_components) == 0, 'double batchify'
        # put the output component to 1 ("batchified")
        return 1, (self, builtin_identity, builtin_identity), True

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode != 'infer':
            return args
        if isinstance(args, (tuple, list)):
            return tuple([self(x) for x in args])
        if isinstance(args, dict):
            return {k: self(args[k]) for k in args}
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim)
        if isinstance(args, (float, int)):
            # pylint: disable=not-callable
            return torch.tensor([args])
        raise TypeError('only tensors and tuples of tensors recursively supported...')


def save(transform: Transform, path: Union[Path, str], force_overwrite: bool = False,
         compress: bool = False):
    """Save the transform to a folder at *path* or a compressed (zip-)file of the same name if
    *compress* == True.

    The folder's name should end with '.padl'. If no extension is given, it will be added
    automatically.

    If the folder exists, call with *force_overwrite* = `True` to overwrite. Otherwise, this
    will raise a FileExistsError.
    """
    if compress:
        transform.pd_zip_save(path, force_overwrite)
    else:
        transform.pd_save(path, force_overwrite)


def load(path):
    """Load a transform (as saved with padl.save) from *path*. """
    if Path(path).is_file():
        return _zip_load(path)
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
    # pylint: disable=no-member,protected-access
    transform = module._pd_main
    for i, subtrans in enumerate(transform._pd_all_transforms()):
        subtrans.pd_post_load(path, i)
    return transform


def _zip_load(path: Union[Path, str]):
    """Load a transform from a compressed '.padl' file. """
    # we can't use TemporaryDirectory with a context because the files need to exist when
    # using / saving again
    dirname = TemporaryDirectory('.padl').name
    with ZipFile(path, 'r') as zipf:
        zipf.extractall(dirname)
        return load(dirname)


def group(transform: Union[Rollout, Parallel]):
    """Group transforms. This prevents them from being flattened when used

    Example:

    When writing a Rollout as `(a + (b + c))`, this is automatically flattened to `(a + b + c)`
    - i.e. the resulting Rollout transform expects a 3-tuple whose inputs are passed to `a`, `b`,
    `c` respectively. To prevent that, do (a + group(b + c)). The resulting Rollout will expect a
    2-tuple whose first item will be passed to `a` and whose second item will be passed to `b + c`.
    """
    return transform.grouped()


class _ItemGetter:
    """A simple item getter. Takes *samples* and applies *transform* to it.

    Example:

    >>> from padl import transform
    >>> ig = _ItemGetter([1, 2, 3], transform(lambda x: x + 1))
    >>> len(ig)
    3
    >>> ig[0]
    2
    >>> ig[1]
    3

    :param samples: An object implementing __getitem__ and __len__.
    :param transform: Preprocessing transform.
    :param exception: Exception to catch for (fall back to *default*).
    :param default: The default value to fall back to in case of exception.
    """

    def __init__(self, samples, transform, exception=None, default=None):
        self.samples = samples
        self.transform = transform
        self.exception = exception
        self.default = default

    def __getitem__(self, item):
        if self.exception:
            try:
                return self.transform(self.samples[item])
            except self.exception:
                return self.default
        return self.transform(self.samples[item])

    def __len__(self):
        return len(self.samples)
