"""The Transform class and its fundamental children.

Transforms should be created using the `padl.transform` wrap-function.
"""

import re
from copy import copy
from collections import Counter, namedtuple, OrderedDict, defaultdict
from functools import lru_cache
import inspect
from itertools import chain
from pathlib import Path
from os import remove
from shutil import rmtree
import textwrap
import traceback
from tempfile import TemporaryDirectory
import types
from dataclasses import dataclass


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple, Union, Any
from warnings import warn
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader

from padl.dumptools import symfinder, inspector
from padl.dumptools.var2mod import CodeGraph, CodeNode, find_codenode
from padl.dumptools.symfinder import ScopedName
from padl.dumptools.serialize import Serializer

from padl.dumptools.packagefinder import dump_packages_versions
from padl.exceptions import WrongDeviceError
from padl.print_utils import combine_multi_line_strings, create_reverse_arrow, make_bold, \
    make_green, create_arrow, format_argument, visible_len

from IPython.display import Image
import networkx as nx


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


class _GeneratorWithLength:
    """A generator with a length. """

    def __init__(self, generator, length):
        self.length = length
        self.generator = generator

    def __iter__(self):
        yield from self.generator()

    def __len__(self):
        return self.length


class _OutputSlicer:
    """Helper class to store output slice"""

    def __init__(self, main_object):
        self.main_object = main_object

    def __getitem__(self, item):
        named_copy = copy(self.main_object)
        named_copy._pd_name = self.main_object._pd_name
        named_copy._pd_group = True
        named_copy._pd_varname = {}
        named_copy.pd_output = _OutputSlicer(named_copy)
        named_copy.pd_input = _InputSlicer(named_copy)
        named_copy._pd_output_slice = item
        return named_copy


class _InputSlicer:
    """Helper class to store input slice"""

    def __init__(self, main_object):
        self.main_object = main_object

    def __getitem__(self, item):
        named_copy = copy(self.main_object)
        named_copy._pd_name = self.main_object._pd_name
        named_copy._pd_group = True
        named_copy._pd_varname = {}
        named_copy.pd_output = _OutputSlicer(named_copy)
        named_copy.pd_input = _InputSlicer(named_copy)
        named_copy._pd_input_slice = item

        return named_copy


Mode = Literal['infer', 'eval', 'train']
Stage = Literal['preprocess', 'forward', 'postprocess']


class Transform:
    """Transform base class.

    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: name of the transform.
    """
    pd_mode = None
    _pd_external_full_dump_modules = set()

    def __init__(self, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        if call_info is None:
            call_info = inspector.CallInfo()
        self._pd_call_info = call_info
        self._pd_varname = {}
        self._pd_name = pd_name
        self._pd_device = 'cpu'
        self._pd_traceback = traceback.extract_stack()
        self._pd_external_full_dump = False
        self._pd_output_slice = None
        self._pd_input_slice = None
        self.pd_output = _OutputSlicer(self)
        self.pd_input = _InputSlicer(self)

    @property
    def _pd_full_dump_relevant_module(self):
        return self._pd_call_info.scope.module

    @property
    def _pd_full_dump(self) -> bool:
        """If *True*, dump the Transform in full (with definition etc) even if it was defined in
        a different module. Else, only dump an import statement. """
        module = self._pd_full_dump_relevant_module
        # always fully dump Transforms from the module the dump was triggered in
        if inspector.caller_module() == module or getattr(module, '__name__', '__main__') == '__main__':
            return True
        # fully dump all Transforms from packages or modules specified in
        # _pd_external_full_dump_modules
        if any(module.__spec__.name.startswith(mod)
               for mod in self._pd_external_full_dump_modules):
            return True
        return self._pd_external_full_dump

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

    @property
    @lru_cache(maxsize=128)
    def pd_stages(self):
        """Get a tuple of the pre-process, forward, and post-process stages."""
        _, splits, has_batchify = self._pd_splits()
        if has_batchify:
            preprocess, forward, postprocess = splits
        else:
            preprocess, forward, postprocess = identity, splits[0], splits[2]
        return preprocess, forward, postprocess

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
                                                      Tuple['Transform',
                                                            'Transform',
                                                            'Transform'],
                                                      bool]:
        """Split the transform into "pre-batchified", "batchified" and "postprocessing" splits.

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
        assert isinstance(component, int), ('A normal Transform cannot process input from multiple '
                                            'stages.')
        return (
            # a normal transform doesn't change the components
            component,
            # for the component the transform is in, return the transform, else Identity
            tuple(self if i == component else identity for i in range(3)),
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
        """Rollout with *other*.

        Example:
            t = a + b + c
        """
        return Rollout([self, other])

    def __truediv__(self, other: "Transform") -> "Parallel":
        """Parallel with *other*.

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
        named_copy._pd_group = True
        named_copy._pd_varname = {}
        named_copy.pd_output = _OutputSlicer(named_copy)
        named_copy.pd_input = _InputSlicer(named_copy)
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

    def _pd_codegraph_add_startnodes(self, graph, name: Union[str, None]) -> Set:
        """Build the start-:class:`CodeNode` objects - the node with the source needed to create
        *self* as *name* (in the scope where *self* was originally created).

        Returns a set of dependencies (scoped names the start-node depends on).
        """
        scope = self._pd_call_info.scope

        nodes = []

        start_source = f'{name or "_pd_dummy"} = {self._pd_evaluable_repr()}'
        start = CodeNode.from_source(start_source, scope)
        nodes.append(start)

        # if name is given, add the node to the CodeGraph, otherwise only use the dependencies
        if name is not None:
            graph[ScopedName(name, scope, 0)] = start

        dependencies = set()
        for node in nodes:
            dependencies.update(node.globals_)
        return dependencies

    @property
    def _pd_closurevars(self) -> Tuple[dict, dict]:
        """Return the closurevars (globals and nonlocals) the transform depends on. """
        return {}, {}

    def _pd_build_codegraph(self, graph: Optional[dict] = None,
                            name: Optional[str] = None) -> Tuple[dict, dict]:
        """Build a codegraph defining the transform.

        A codegraph's nodes are :class:`CodeNode` instances which contain a scoped name, a piece of
        code defining the name and a set of dependencies (other scoped names). The dependencies
        can be understood as the edges_dict in the graph.

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
        :param name: The name to give the transform.
        :return: Updated graph.
        """
        if graph is None:
            graph = CodeGraph()

        # build the start node ->
        # if the *name* is the same as the call, we don't need to assign to the name
        # this can be the case for function transforms
        if getattr(self, '_pd_call', None) == name:
            new_name = None
        else:
            new_name = name

        todo = self._pd_codegraph_add_startnodes(graph, new_name)

        # if this transform has closurevars, get them (if there are transforms in the closure, we
        # want to allow them to build their codegraph themselves, see below)
        globals_dict, nonlocals_dict = self._pd_closurevars
        all_vars_dict = {**globals_dict, **nonlocals_dict}

        # find dependencies
        while todo:
            next_name = todo.pop()

            # we know this already - go on
            if next_name in graph:
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
                next_obj._pd_build_codegraph(graph, next_name.name)
            except (KeyError, AttributeError):
                pass
            else:
                continue

            # find how *next_name* came into being
            next_codenode = find_codenode(next_name)
            graph[next_name] = next_codenode

            todo.update(next_codenode.globals_)

        return graph

    def _pd_process_traceback(self):
        """Find where the Transform was defined (file, lineno, file) given the traceback. """
        a_tb = None
        for a_tb in self._pd_traceback[::-1]:
            if 'padl' in a_tb[0]:
                continue
            break
        return f'{a_tb.filename} in {a_tb.name}\n----> {make_green(a_tb.lineno)}    {a_tb.line}'

    def _pd_get_error_idx(self):
        """Get what element of a :class:`padl.transforms.Transform` is failing if an Exception is
        produced during an execution.

        Subclasses of :class:`padl.transforms.Transform` need to implement this method.
        """
        return NotImplemented

    def _pd_trace_error(self, position: int, arg):
        """Add some error description to :obj:`pd_trace`. """
        try:
            str_ = self._pd_fullrepr(marker=(position, '\033[31m  <---- error here \033[0m'))
            _pd_trace.append(_TraceItem(str_, self._pd_process_traceback(), arg,
                                        self, Transform.pd_mode, position))
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

        This includes the transform itself, the subtransforms of a pipeline transform or
        transforms a function-transform depends on as a global."""
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
        graph = self._pd_build_codegraph(name='_pd_main')
        Serializer.save_all(graph, path)
        code = graph.dumps()
        if return_versions:
            versions = dump_packages_versions(node.ast_node for node in graph.values())
            return code, versions
        return code

    def __repr__(self):
        return self._pd_shortrepr(formatting=False)

    def _repr_pretty_(self, p, cycle):
        #pylint: disable=invalid-name
        p.text(self._pd_fullrepr() if not cycle else '...')

    def _pd_fullrepr(self, marker=None):
        title = self._pd_title()
        if self.pd_name is not None and self.pd_name != title:
            title = make_bold(title) + f' - "{self.pd_name}"'
        else:
            title = make_bold(title)
        top_message = title + ':' + '\n\n'
        bottom_message = textwrap.indent(self._pd_longrepr(marker=marker), '   ')
        return top_message + bottom_message

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        """A line string representation of the transform."""
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

    def pd_varname(self, scope=None) -> Optional[str]:
        """The name of the variable name the transform was last assigned to.

        Example:

        >>> from padl import transform
        >>> foo = transform(lambda x: x + 1)
        >>> foo.pd_varname()  # doctest: +SKIP
        'foo'

        :param scope: Scope to search
        :return: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        if scope is None:
            scope = self._pd_call_info.scope
        if scope not in self._pd_varname:
            self._pd_varname[scope] = self._pd_find_varname(scope.module.__dict__)
        return self._pd_varname[scope]

    def pd_forward_device_check(self) -> bool:
        """Check if all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole Pipeline.
        """
        pd_device = self.pd_device
        for layer in self.pd_forward.pd_layers:
            for parameters in layer.parameters():
                parameter_device = parameters.device.type
                if ':' in pd_device and 'cuda' in parameter_device:
                    parameter_device += f':{parameters.device.index}'
                if parameter_device != pd_device:
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
            parameters = self._pd_signature.values()
        except ValueError:
            return False
        for param in parameters:
            param_kind = param.kind
            if param_kind in (
                    param.POSITIONAL_OR_KEYWORD,
                    param.POSITIONAL_ONLY):
                signature_count += 1
            if param_kind == param.VAR_POSITIONAL:
                return True
        if signature_count > 1:
            return True
        return False

    def _pd_unpack_args_and_call(self, arg):
        try:
            if self._pd_unpack_argument(arg):
                return self(*arg)
            return self(arg)
        except Exception as err:
            self._pd_trace_error(0, arg)
            raise err

    def pd_call_in_mode(self, arg, mode: Mode, ignore_grad=False):
        """Call the transform, with possibility to pass multiple arguments.

        :param arg: argument to call the transform with
        :param mode: The mode ("infer", "eval", "train") to perform the call with.
        :param ignore_grad: If *True* gradient settings are ignored
        :return: Whatever the transform returns.
        """
        no_grad = mode in ('eval', 'infer') and not ignore_grad

        layers = self.pd_layers
        if not layers:
            no_grad = False

        if mode is not None:
            Transform.pd_mode = mode
            if layers:
                training_before = [layer.training for layer in layers]
                for layer in layers:
                    if mode == 'train':
                        layer.train()
                    else:
                        layer.eval()

        if no_grad:
            grad_before = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        try:
            return self._pd_unpack_args_and_call(arg)
        finally:
            if mode is not None:
                Transform.pd_mode = None
                if layers:
                    for i, training in enumerate(training_before):
                        layer = layers[i]
                        if training:
                            layer.train()
                        else:
                            layer.eval()
            if no_grad:
                torch.set_grad_enabled(grad_before)

    @property
    @lru_cache(maxsize=128)
    def _pd_signature(self):
        """Get the signature of the transform. """
        return inspect.signature(self).parameters

    def _pd_format_output(self, x):
        return x

    def _pd_itercall(self, args, mode: Mode, loader_kwargs: Optional[dict] = None,
                     flatten: bool = False) -> Iterator:
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param mode: Mode to call in ("eval", "train" or "infer")
        :param loader_kwargs: Data loader keyword arguments.
        :param flatten: If *True*, flatten the output.

        :return: A generator that allows iterating over the output.
        """
        assert mode in ('eval', 'train'), '_pd_itercall can only be used with mode eval or train'

        _pd_trace.clear()

        self.pd_forward_device_check()

        preprocess = self.pd_preprocess
        forward = self.pd_forward
        post = self.pd_postprocess

        use_preprocess = not isinstance(preprocess, Identity)
        use_forward = not isinstance(forward, Identity)
        use_post = not isinstance(post, Identity)

        if use_forward:
            self.pd_forward_device_check()

        if use_preprocess:
            loader = self.pd_get_loader(args, preprocess, mode, **loader_kwargs)
        else:
            loader = _SimpleGetter(args)

        def _gen():
            for ix, batch in loader:
                batch = _move_to_device(batch, self.pd_device)
                output = batch

                if use_forward:
                    try:
                        output = forward.pd_call_in_mode(batch, mode)
                    except Exception as err:
                        self._pd_trace_error(self._pd_get_error_idx('forward'),
                                             [args[i] for i in ix])
                        raise err

                if use_post or flatten:
                    output = _unpack_batch(output)
                    if use_post:
                        try:
                            output = [post.pd_call_in_mode(x, mode, ignore_grad=True) \
                                      for x in output]
                        except Exception as err:
                            self._pd_trace_error(self._pd_get_error_idx('postprocess'),
                                                 [args[i] for i in ix])
                            raise err

                    for out in output:
                        yield self._pd_format_output(out)
                else:
                    yield self._pd_format_output(output)

        if use_post or flatten:
            length = len(args)
        else:
            length = len(loader)

        return _GeneratorWithLength(_gen, length)

    @property
    def pd_device(self) -> str:
        """Return the device ("cpu" / "cuda") the transform is on."""
        return self._pd_device

    @pd_device.setter
    def pd_device(self, device: str):
        self._pd_device = device

    @property
    def pd_preprocess(self) -> "Transform":
        """The preprocessing part of the transform. The device must be propagated from self."""
        pre = self.pd_stages[0]
        pre.pd_to(self.pd_device)
        return pre

    @property
    def pd_forward(self) -> "Transform":
        """The forward part of the transform (that what's typically done on the GPU).
        The device must be propagated from self."""
        forward = self.pd_stages[1]
        forward.pd_to(self.pd_device)
        return forward

    @property
    def pd_postprocess(self) -> "Transform":
        """The postprocessing part of the transform. The device must be propagated from self."""
        post = self.pd_stages[2]
        post.pd_to(self.pd_device)
        return post

    def pd_to(self, device: str) -> "Transform":
        """Set the transform's device to *device*.

        :param device: Device to set the transform to {'cpu', 'cuda', 'cuda:N'}.
        """
        self.pd_device = device
        for layer in self.pd_layers:
            layer.to(device)
        return self

    @property
    @lru_cache(maxsize=128)
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

    def pd_get_loader(self, args, preprocess, mode, **kwargs) -> DataLoader:
        """Get a pytorch data loader.

        :param args: A sequence of datapoints.
        :param preprocess: preprocessing step
        :param mode: mode
        :param kwargs: Keyword arguments passed to the data loader (see the pytorch
            `DataLoader` documentation for details).
        """
        sequence = _ItemGetter(
            args,
            lambda *args: preprocess.pd_call_in_mode(*args, mode, ignore_grad=True),
            self
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
        in_args = inputs
        _pd_trace.clear()

        preprocess = self.pd_preprocess
        forward = self.pd_forward
        postprocess = self.pd_postprocess
        pd_device = self.pd_device

        use_preprocess = not isinstance(preprocess, Identity)
        use_forward = not isinstance(forward, Identity)
        use_post = not isinstance(postprocess, Identity)

        if use_forward:
            self.pd_forward_device_check()

        if use_preprocess:
            inputs = preprocess.pd_call_in_mode(inputs, mode='infer', ignore_grad=True)
        if pd_device != 'cpu':
            inputs = _move_to_device(inputs, pd_device)
        if use_forward:
            try:
                inputs = forward.pd_call_in_mode(inputs, mode='infer')
            except Exception as err:
                self._pd_trace_error(self._pd_get_error_idx('forward'), in_args)
                raise err
        if use_post:
            try:
                inputs = postprocess.pd_call_in_mode(inputs, mode='infer', ignore_grad=True)
            except Exception as err:
                self._pd_trace_error(self._pd_get_error_idx('postprocess'), in_args)
                raise err
        return self._pd_format_output(inputs)

    def eval_apply(self, inputs: Iterable, flatten: bool = False, **kwargs):
        """Call transform within the eval context.

        This will use multiprocessing for the preprocessing part via `DataLoader` and turn
        of gradients for the forward part.

        It expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_itercall(inputs, 'eval', loader_kwargs=kwargs,
                                 flatten=flatten)

    def train_apply(self, inputs: Iterable, flatten: bool = False, **kwargs):
        """Call transform within the train context.

        This will use multiprocessing for the preprocessing part via `DataLoader` and turn
        on gradients for the forward part.

        It expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_itercall(inputs, 'train', loader_kwargs=kwargs, flatten=flatten)


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

    def _pd_get_non_target_stage_idx(self):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
            t = forward_1 >> unbatch >> post

            Let's suppose we get an error on `post`, `post` is the index 1 of
            `t.pd_postprocess`, then the element that fails on `t` is
            t.pd_forward._pd_get_non_target_stage_idx() + 1 = 1 + 1 = 2
        """
        return 1

    def _pd_get_target_stage_idx(self, is_entire_transform=None):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
            t = prep >> batch >> forward_1
            Let's suppose we get an error on `forward_1`. `forward_1` is the index 0 of
            `t.pd_forward`, then the element that fails on `t` is
            2 + t.pd_forward._pd_get_stage_idx() = 2 + 0 = 2

        :param is_entire_transform: *False* if *self* is not a part of a larger :class:`Transform`,
            else *True*
        """
        return 0

    def _pd_get_error_idx(self, stage: str):
        assert stage in ('forward', 'postprocess')
        return 0


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
    :param inline_wrap: True if the function was wrapped in-line.
    """

    def __init__(self, function: Callable, call_info: inspector.CallInfo,
                 pd_name: Optional[str] = None, call: Optional[str] = None,
                 source: Optional[str] = None, wrap_type: str = 'decorator'):
        if call is None:
            call = function.__name__
        super().__init__(call=call, call_info=call_info, pd_name=pd_name)
        self.function = function
        self._pd_number_of_inputs = None
        self._source = source
        self._wrap_type = wrap_type

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        if not self._pd_full_dump and self._wrap_type == 'inline':
            return f'transform({self.__name__})'
        return self._pd_call

    def _pd_codegraph_add_startnodes(self, graph, name):
        if self._pd_full_dump or self._wrap_type in ('module', 'lambda'):
            return super()._pd_codegraph_add_startnodes(graph, name)
        module = inspector.caller_module()
        scope = symfinder.Scope.toplevel(module)
        source = f'from {self.__module__} import {self.__name__}'

        if self._wrap_type == 'inline':
            node = CodeNode.from_source(source, scope)
            graph[ScopedName(self.__name__, scope, 0)] = node
            emptyscope = symfinder.Scope.empty()
            graph[ScopedName('transform', emptyscope, 0)] = \
                CodeNode.from_source('from padl import transform', emptyscope)

            start_source = f'{name or "_pd_dummy"} = transform({self.__name__})'
            start = CodeNode.from_source(start_source, scope)
            if name is not None:
                graph[ScopedName(name, scope, 0)] = start
            return {}

        if name is not None:
            source += f' as {name}'
        else:
            name = self.__name__
        node = CodeNode.from_source(source, scope)
        graph[ScopedName(name, scope, 0)] = node
        return {}

    @property
    def source(self) -> str:
        """The source of the wrapped function. """
        if self._source is not None:
            return self._source
        body_msg = inspect.getsource(self.function)
        body_msg = ''.join(re.split('(def )', body_msg, 1)[1:])
        return body_msg

    @property
    @lru_cache(maxsize=128)
    def _pd_signature(self) -> List[str]:
        if self._pd_number_of_inputs is None:
            return inspect.signature(self).parameters
        return [f'arg_{i}' for i in range(self._pd_number_of_inputs)]

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        try:
            str_ = self.source.split('\n')[:30]
            if marker:
                return str_[0] + marker[1] + '\n'.join(str_[1:])
            return '\n'.join(str_)
        except TypeError:
            return self._pd_call + marker[1] + '\n' if marker else self._pd_call

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
    def _pd_full_dump_relevant_module(self):
        return inspect.getmodule(self.__class__)

    def _pd_codegraph_add_startnodes_import_var(self, graph, name):
        instance_scope = self._pd_call_info.scope
        varname = self.pd_varname(instance_scope)

        import_source = f'from {self.__module__} import {varname}'
        import_node = CodeNode.from_source(import_source, instance_scope)

        graph[ScopedName(varname, instance_scope, 0)] = import_node

        if name != varname:
            start_source = f'{name or "_pd_dummy"} = {varname}'
            start_node = CodeNode.from_source(start_source, instance_scope)
            if name is not None:
                graph[ScopedName(name, instance_scope, 0)] = start_node

        return set()

    def _pd_codegraph_add_startnodes_import(self, graph, name):
        instance_scope = self._pd_call_info.scope

        # instance creation and class definition are in separate modules
        if instance_scope.module_name != self.__class__.__module__:
            return super()._pd_codegraph_add_startnodes(graph, name)

        # the instance has a varname - just import the instance
        if self.pd_varname(instance_scope) is not None:
            return self._pd_codegraph_add_startnodes_import_var(graph, name)

        # import the class
        import_source = f'from {self.__class__.__module__} import {self.__class__.__name__}'
        import_node = CodeNode.from_source(import_source, instance_scope)
        graph[ScopedName(self.__class__.__name__, instance_scope, 0)] = import_node
        nodes = [import_node]

        # make the call
        call = self.__class__.__name__ + f'({self._split_call()[1]})'
        call_scope = symfinder.Scope.toplevel(inspector.caller_module())
        start_source = f'{name or "_pd_dummy"} = {call}'
        start_node = CodeNode.from_source(start_source, call_scope)
        if name is not None:
            graph[ScopedName(name, call_scope, 0)] = start_node
        nodes.append(start_node)

        dependencies = set()
        for node in nodes:
            dependencies.update(node.globals_)
        return dependencies

    def _pd_codegraph_add_startnodes_full(self, graph, name):
        call_scope = self._pd_call_info.scope
        class_scope = self._pd_class_call_info.scope
        if class_scope == call_scope:
            return super()._pd_codegraph_add_startnodes(graph, name)

        call = self.__class__.__name__ + f'({self._split_call()[1]})'
        start_source = f'{name or "_pd_dummy"} = {call}'
        start_node = CodeNode.from_source(start_source, call_scope)
        if name is not None:
            graph[ScopedName(name, call_scope, 0)] = start_node

        for scoped_name in start_node.globals_:
            if scoped_name.name == self.__class__.__name__:
                scoped_name.scope = class_scope

        return set(start_node.globals_)

    def _pd_codegraph_add_startnodes(self, graph, name):
        if self._pd_full_dump:
            return self._pd_codegraph_add_startnodes_full(graph, name)
        return self._pd_codegraph_add_startnodes_import(graph, name)

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

    def _pd_longrepr(self, marker=None) -> str:
        try:
            str_ = self.source.split('\n')[:30]
            if marker:
                return str_[0] + marker[1] + '\n' + '\n'.join(str_[1:])
            return '\n'.join(str_)
        except symfinder.NameNotFound:
            return self._pd_call + marker[1] if marker else self._pd_call

    def _pd_title(self) -> str:
        title = type(self).__name__
        return title + '(' + self._formatted_args() + ')'


class TorchModuleTransform(ClassTransform):
    """Transform class for use with `torch.nn.Module`."""

    @property
    @lru_cache(maxsize=128)
    def _pd_signature(self):
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

    def _pd_longrepr(self, marker=None) -> str:
        out = torch.nn.Module.__repr__(self)
        if marker:
            return out + marker[1]
        return out


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
                tuple(Map(split) if not isinstance(split, Identity) else identity
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
            tuple(Parallel(s) if s else identity for s in splits),
            has_batchify
        )

    def __call__(self, args: Iterable):
        """
        :param args: Args list to call transforms with
        """
        return tuple([self.transform._pd_unpack_args_and_call(arg) for arg in args])

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        str_ = '~ ' + self.transform._pd_shortrepr(formatting)
        return str_ + marker[1] if marker else str_

    @property
    def _pd_direct_subtransforms(self) -> Iterator[Transform]:
        yield self.transform

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        varname = self.transform.pd_varname()
        if varname:
            return f'~{varname}'
        return f'~{self.transform._pd_evaluable_repr(indent)}'


class Pipeline(Transform):
    """Abstract base class for Pipeline

    :param transforms: List of sub-transforms.
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    :param pd_name: Name of the Pipeline.
    :param pd_group: If *True*, do not flatten this when used as child transform in a
        :meth:`Pipeline`.
    """
    op = NotImplemented
    display_op = NotImplemented

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(call_info, pd_name)

        self._pd_group = True if pd_name is not None else pd_group

        transforms = self._flatten_list(transforms)
        self.transforms: List[Transform] = transforms

    def _pd_format_output(self, x):
        try:
            return self.transforms[-1]._pd_output_formatter(x)
        except AttributeError:
            return x

    def __sub__(self, name: str) -> "Transform":
        """Create a named clone of the transform.

        Example:
            named_t = t - 'rescale image'
        """
        named_copy = copy(self)
        named_copy._pd_name = name
        named_copy._pd_group = True
        named_copy._pd_varname = {}
        named_copy.pd_output = _OutputSlicer(named_copy)
        named_copy.pd_input = _InputSlicer(named_copy)
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

    def _pd_unpack_args_and_call(self, arg):
        if self._pd_unpack_argument(arg):
            return self(*arg)
        return self(arg)

    def _pd_evaluable_repr_inner(self, indent=0):
        sub_reprs = [
            x.pd_varname(self._pd_call_info.scope) or x._pd_evaluable_repr(indent + 4)
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

    def _defined_somewhere_else(self):
        defined_as = self.pd_varname(self._pd_call_info.scope)
        return (
                self._pd_call_info.scope.module_name != inspector.caller_module().__name__
                and defined_as is not None
        )

    def _codegraph_add_import_startnode(self, graph, name):
        module = inspector.caller_module()
        scope = symfinder.Scope.toplevel(module)
        defined_as = self.pd_varname(self._pd_call_info.scope)
        source = f'from {self._pd_call_info.scope.module_name} import {defined_as}'
        if name is not None and name != defined_as:
            source += f' as {name}'
        else:
            name = defined_as
        node = CodeNode.from_source(source, scope)
        graph[ScopedName(name, scope, 0)] = node

    def _pd_build_codegraph(self, graph=None, name=None):
        """Build a codegraph defining the transform.

        See :meth:`Transform._pd_build_codegraph` for an explanation of what a code-graph is.

        The codegraph of a :class:`Pipeline` is the union of the codegraphs of the
        contained transforms plus the node defining the transform itself.
        """
        if graph is None:
            graph = CodeGraph()

        if not self._pd_full_dump and self._defined_somewhere_else():
            self._codegraph_add_import_startnode(graph, name)
            return graph

        self._pd_codegraph_add_startnodes(graph, name)

        if self._pd_group and 'padl' not in graph:
            emptyscope = symfinder.Scope.empty()
            graph[ScopedName('padl', emptyscope, 0)] = CodeNode.from_source('import padl',
                                                                            emptyscope)

        # iterate over sub-transforms and update the codegraph with their codegraphs
        for transform in self.transforms:
            varname = transform.pd_varname(self._pd_call_info.scope)
            # pylint: disable=protected-access
            transform._pd_build_codegraph(graph, varname)
        return graph

    def _pd_longrepr(self, formatting=True, marker=None):
        between = f'\n{make_green(self.op, not formatting)}  \n'
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
        self.pd_device = device
        for transform_ in self.transforms:
            transform_.pd_to(device)
        return self

    def pd_forward_device_check(self):
        """Check all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole Pipeline.

        :return: Bool
        """
        if isinstance(self.pd_forward, Identity):
            return True

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
                name = 'out_' + str(ind)
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

    def _pd_get_non_target_stage_idx(self):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
            t = ((prep_1 >> batch) + (prep_2 >> batch)) >> forward_1
            Let's suppose we get an error on `forward_1`. `forward_1` is the index 0 of
            `t.pd_forward`, then the element that fails on `t` is
            t.pd_preprocess._pd_get_non_target_stage_idx() + 0 = 1 + 0 = 1
        """
        return 1

    def _pd_get_target_stage_idx(self, is_entire_transform=None):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
            t = prep >> batch >> (forward_1 + forward_2)
            Let's suppose we get an error on `(forward_1 + forward_2)`. `(forward_1 + forward_2)`
            is the index 0 of `t.pd_forward`, then the element that fails on `t` is
            2 + t.pd_forward._pd_get_stage_idx() = 2 + 0 = 2

        :param is_entire_transform: *False* if *self* is a part of a larger :class:`Transform`,
            else *True*
        """

        return 0

    def _pd_get_error_idx(self, stage: str):
        """Track the index where a :class:`Pipeline` fails from the one that got the Exception on
        :meth:`self.pd_preprocess`, :meth:`self.pd_forward` or :meth:`self.pd_postprocess`.

        Example:
            t = (t_11 >> batch >> t_12) + (t_21 >> batch >> t_22)
            then,
            t.pd_forward = t_12 / t_22.
            If we know that :meth:`t.pd_forward` is failing on `t_22`, which is its element 1,
            the index of the :class:`Transform` that fails on `t` is the same.
        """
        assert stage in ('forward', 'postprocess')
        return _pd_trace[-1].error_position

    def _add_name_to_splits(self, final_splits):
        """Add name to split-transforms. """
        if self._pd_name is not None:
            for i, s in enumerate(final_splits):
                if not isinstance(s, Identity):
                    final_splits[i] = s - self._pd_name


class Compose(Pipeline):
    """Apply series of transforms on input.

    Compose([t1, t2, t3])(x) = t3(t2(t1(x)))

    :param transforms: List of transforms to compose.
    :param call_info: A `CallInfo` object containing information about how the transform was
        created (needed for saving).
    :param pd_name: name of the Compose transform.
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
                if isinstance(split[0], Compose):
                    final_splits.append(group(split[0]))
                else:
                    final_splits.append(split[0])
            else:  # if it's empty: identity
                final_splits.append(identity)

        self._add_name_to_splits(final_splits)
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

    def _pd_longrepr(self, formatting=True, marker=None) -> str:  # TODO: make it respect the formatting
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
                rows[i] = f' {make_green(self.transforms[i].op)} '.join(r)
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
                    all_params.append(list(tt._pd_signature.keys()))
                to_combine = [
                    ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + tuple_to_str(params)
                    if len(params) > 1
                    else ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + params[0]
                    for k, params in enumerate(all_params)
                ]
                to_format = combine_multi_line_strings(to_combine)
            else:
                params = t._pd_signature
                to_format = '  ' + tuple_to_str(params) if len(params) > 1 else '  ' + \
                                                                                list(params)[0]
            to_format_pad_length = max([len(x.split('\n')) for x in subarrows]) - 1
            to_format = ''.join(['\n' for _ in range(to_format_pad_length)] + [to_format])

            # combine the arrows
            mark = combine_multi_line_strings(subarrows + [to_format])
            mark = '\n'.join(['   ' + x for x in mark.split('\n')])
            output.append(make_green(mark))
            output.append(make_bold(f'{i}: ') + r + (marker[1] if marker and
                                                                  marker[0] == i else ''))
        return '\n'.join(output)

    def __call__(self, args):
        """Call method for Compose

        :param args: Arguments to call with.
        :return: Output from series of transforms.
        """
        _in_args = args
        for i, transform_ in enumerate(self.transforms):
            try:
                args = transform_._pd_unpack_args_and_call(args)
            except Exception as err:
                self._pd_trace_error(i, _in_args)
                raise err
        return args

    def _pd_get_non_target_stage_idx(self):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
            t = prep >> batch >> forward_1

            Let's suppose we get an error on `forward_1`, `forward_1` is the index 0 of
            `t.pd_forward`, then the element that fails on `t` is
            `t.pd_preprocess._pd_get_non_target_stage_idx() + 0 = len(t.pd_preprocess) + 0 =
                2 + 0 = 2
        """
        return 1 if self._pd_group else len(self)

    def _pd_get_target_stage_idx(self, is_entire_transform: bool):
        """Return an integer to track where a :class:`Compose` which failed got the Exception.

        Example:
             t = prep >> batch >> forward_1 >> forward_2

            Let's suppose we get an error on `forward_2`. `t.pd_forward` is
            `t.pd_forward` = forward_1 >> forward_2`. `forward_2` is the index 1 of `t.pd_forward`,
            then the element that fails on `t` is
            2 + t.pd_forward._pd_get_stage_idx() = 2 + 1 = 3

        :param is_entire_transform: *False* if *self* is a part of a larger :class:`Transform`,
            else *True*
        """
        return 0 if self._pd_group and not is_entire_transform else _pd_trace[-1].error_position

    def _pd_get_error_idx(self, stage: str):
        """Track the index where a :class:`Compose` fails from the index that got the Exception
        on :meth:`self.pd_preprocess`, :meth:`self.pd_forward` or :meth:`self.pd_postprocess`.

        Examples:
            t = t_1 >> t_2 >> batch >> t_3 >> t_4
            then,
            t.pd_forward = t_3 >> t_4.
            If we know that :meth:`t.pd_forward` is failing on `t_4`, which is its element 1, then
            `t` is failing on len(t.pd_preprocess) + 1.

            t = t_1 >> t_2 >> batch >> ((t_3 >> t_4) + (t_5 >> t_6))
            then,
            t.pd_forward = (t_3 >> t_4) + (t_5 >> t_6).
            No matter what branch is failing on :meth:`t.pd_forward`, the error on `t` is on
            len(t.pd_preprocess) + 0 = 3.
        """
        assert stage in ('forward', 'postprocess')
        preprocess = self.pd_preprocess
        forward = self.pd_forward
        postprocess = self.pd_postprocess
        preprocess_idx = preprocess._pd_get_non_target_stage_idx()

        if stage == 'forward':
            is_entire_transform = isinstance(preprocess, Identity) and \
                                  isinstance(postprocess, Identity)
            return preprocess_idx + forward._pd_get_target_stage_idx(is_entire_transform)

        is_entire_transform = isinstance(preprocess, Identity) and \
                              isinstance(forward, Identity)
        return preprocess_idx + forward._pd_get_non_target_stage_idx() + \
               postprocess._pd_get_target_stage_idx(is_entire_transform)


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
        self._pd_output_formatter = lambda x: namedtuple('namedtuple', self.pd_keys)(*x)

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
            # pylint: disable=protected-access
            sub_output_components, sub_splits, sub_has_batchify = \
                transform_._pd_splits(input_components)
            has_batchify = has_batchify or sub_has_batchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # only replace with identity if all Identity to preserve number of pipes

        merged_components = self._pd_merge_components(input_components)
        if not isinstance(merged_components, int):
            merged_components = 0

        cleaned_splits = []
        for i, split in enumerate(splits):
            if all(isinstance(s, Identity) for s in split):
                if i != merged_components:
                    cleaned_splits.append(identity)
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

        res = []
        for split in final_splits:
            try:
                res.append(group(split))
            except AttributeError:
                res.append(split)
        final_splits = res

        self._add_name_to_splits(final_splits)
        return output_components, final_splits, has_batchify

    def __call__(self, args):
        """Call method for Rollout

        :param args: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for i, transform_ in enumerate(self.transforms):
            try:
                out.append(transform_._pd_unpack_args_and_call(args))
            except Exception as err:
                self._pd_trace_error(i, args)
                raise err

        if Transform.pd_mode is not None:
            return tuple(out)
        return self._pd_output_formatter(out)

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        make_green_ = lambda x: make_green(x, not formatting)
        make_bold_ = lambda x: make_bold(x, not formatting)
        between = f'\n{make_green_("│ " + self.op)}  \n'
        rows = [make_green_('├─▶ ') + make_bold_(f'{i}: ') + t._pd_shortrepr()
                + (marker[1] if marker and marker[0] == i else '')
                for i, t in enumerate(self.transforms[:-1])]
        rows.append(make_green_('└─▶ ') + make_bold_(f'{len(self.transforms) - 1}: ')
                    + self.transforms[-1]._pd_shortrepr() +
                    (marker[1] if marker and marker[0] == len(self.transforms) - 1 else ''))
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
        self._pd_output_formatter = lambda x: namedtuple('namedtuple', self.pd_keys)(*x)

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

        # only replace with identity if all Identity to preserve number of pipes
        cleaned_splits = tuple(
            identity if all(isinstance(s, Identity) for s in split) else split
            for split in splits
        )

        final_splits = tuple(Parallel(s) if isinstance(s, list) else s for s in cleaned_splits)

        res = []
        for split in final_splits:
            try:
                res.append(group(split))
            except AttributeError:
                res.append(split)
        final_splits = res

        self._add_name_to_splits(final_splits)
        return output_components, final_splits, has_batchify

    def __call__(self, args):
        """Call method for Parallel

        :param args: Argument to call with.
        :return: Namedtuple of output.
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            try:
                out.append(transform_._pd_unpack_args_and_call(args[ind]))
            except Exception as err:
                self._pd_trace_error(ind, args)
                raise err

        if Transform.pd_mode is not None:
            return tuple(out)
        return self._pd_output_formatter(out)

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
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
                make_bold_(f'{i}: ') + t._pd_shortrepr()
            )
            out += marker[1] + '\n' if marker and marker[0] == i else '\n'
            if i < len(self.transforms) - 1:
                out += f'{make_green_(pipes(len_ - i - 1) + spaces(i + 2) + self.op)}  \n'

        return out


class BuiltinTransform(ClassTransform):
    """A builtin transform will simply always be imported, never fully dumped. """

    def _pd_longrepr(self, formatting=True, marker=None):
        out = self._pd_call.split('padl.')[-1]
        if marker:
            out += marker[1]
        return out

    @property
    def _pd_full_dump(self):
        return False


class Identity(BuiltinTransform):
    """Do nothing. Just pass on."""

    def __init__(self):
        super().__init__()

    def __call__(self, args):
        return args

    def _pd_get_non_target_stage_idx(self):
        return 0


identity = Identity()


class Unbatchify(BuiltinTransform):
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
        return 2, (identity, identity, self), False

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode != 'infer':
            return _move_to_device(args, 'cpu') if self.cpu else args
        if isinstance(args, tuple):
            return tuple([self(x) for x in args])
        if isinstance(args, list):
            return [self(x) for x in args]
        if isinstance(args, torch.Tensor):
            args = args.squeeze(self.dim)
            return args.to('cpu') if self.cpu else args

        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(BuiltinTransform):
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
        return 1, (self, identity, identity), True

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
    code = compile(source, path / 'transform.py', 'exec')
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
    :param entire_transform: :class:`Transform` which *transform* belongs to, i.e.,
        :class:`Transform` whose preprocessing part is *transform*.
    """

    def __init__(self, samples, transform, entire_transform):
        self.samples = samples
        self.transform = transform
        self.entire_transform = entire_transform

    def __getitem__(self, item):
        try:
            return item, self.transform(self.samples[item])
        except Exception as err:
            is_entire_transform = isinstance(self.entire_transform.pd_forward, Identity) and \
                                  isinstance(self.entire_transform.pd_postprocess, Identity)
            self.entire_transform._pd_trace_error(
                self.entire_transform.pd_preprocess._pd_get_target_stage_idx(is_entire_transform),
                [self.samples[item]]
            )
            raise err

    def __len__(self):
        return len(self.samples)


class _SimpleGetter:
    """A simple item getter.

    :param samples: An object implementing __getitem__ and __len__.
    """

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, item):
        return [item], self.samples[item]

    def __len__(self):
        return len(self.samples)

@dataclass
class _TraceItem:
    """Catch information of an Exception produced in a Transform call.

    :param transform_str: string representation of a *Transform* that has produced an Exception.
    :param code_position: line where the Transform that has produced the Exception was defined.
    :param args: arguments input to the Transform.
    :param transform: Transform that produced the Exception.
    :param pd_mode: mode (*train*, *eval* or *infer*) of *transform*
    :param error_position: item inside *transform* that has produced the Exception.
    """
    transform_str: str
    code_position: str
    args: Any
    transform: Transform
    pd_mode: str
    error_position: int


def fulldump(transform_or_module):
    """Switch a Transform or module or package to the "fulldump" mode.

    This means that the Transform or any Transform from that module or package will be fully dumped
    instead of just dumping the statement importing it.

    :param transform_or_module: A Transform, module or package for which to enable full dump. Can
        also be a string. In that case, will enable full dump for the module or package with
        matching name.
    """
    if isinstance(transform_or_module, types.ModuleType):
        transform_or_module = transform_or_module.__spec__.name
    if isinstance(transform_or_module, str):
        Transform._pd_external_full_dump_modules.add(transform_or_module)
        return None
    assert isinstance(transform_or_module, Transform)
    t_copy = copy(transform_or_module)
    t_copy._pd_external_full_dump = True
    return t_copy


def importdump(transform_or_module):
    """Disable full dump (see :func:`padl.transforms.fulldump` for more). """
    if isinstance(transform_or_module, types.ModuleType):
        transform_or_module = transform_or_module.__spec__.name
    if isinstance(transform_or_module, str):
        try:
            Transform._pd_external_full_dump_modules.remove(transform_or_module)
        except KeyError:
            pass
        return None
    assert isinstance(transform_or_module, Transform)
    t_copy = copy(transform_or_module)
    t_copy._pd_external_full_dump = False
    return t_copy


class Node:
    """Node wrapper for transforms

    Nodes wrap a transform and can be connected in graph to form complex
    graphs.
    :param transform: Transform to wrap
    """
    _id = 0

    def __init__(self, transform: Transform):
        self.id = self._generate_id()  # keep?
        self.transform = transform
        self.pd_input_slice = self.transform._pd_input_slice
        self.pd_output_slice = self.transform._pd_output_slice

    @property
    def name(self):
        if self.transform.pd_name is None and isinstance(self.transform, Graph):
            return self.transform._name
        return self.transform.pd_name

    @property
    def name_id(self):
        return self.name + ' ' + str(self.id)

    @classmethod
    def _generate_id(cls):
        cls._id += 1
        return cls._id

    def __call__(self, args):  # should a node be callable?
        """Call Method for Node"""
        return self.call_node(args)

    def copy(self):
        _copy = type(self)(self.transform)
        return _copy

    def call_node(self, args):
        return self.transform.pd_call_in_mode(args, mode='infer', ignore_grad=True)


class Graph(Pipeline):
    """Graph: New Pipeline

    * Nodes listed: Should Contain Nodes to do the operation
    * Edges are to be connected and only contained in a Graph
    Example:
    comp1 = A >> B >> C
    comp2 = comp1 >> X >> Y
    the connection from `comp1 >> X` should not leak in comp1
    """

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

        self.input_node = Node(identity - 'Input')  # potential overhead
        self.output_node = Node(identity - 'Output')

        self.networkx_graph = None  # ... later drop

        self._pd_group = False
        self.edges = defaultdict(dict)
        self.parents = defaultdict(list)

    def copy(self):
        """Return copy of this graph"""
        copy_graph = type(self)(self.transforms)
        copy_graph._pd_name = self._pd_name
        copy_graph._pd_group = self._pd_group
        copy_graph._pd_varname = {}
        copy_graph.pd_output = _OutputSlicer(copy_graph)
        copy_graph.pd_input = _InputSlicer(copy_graph)
        return copy_graph

    @classmethod
    def _flatten_list(cls, transform_list: List[Transform]):
        """Flatten *list_* such that members of *cls* are not nested.

        :param transform_list: List of transforms.
        """
        list_flat = []

        for transform in transform_list:
            if isinstance(transform, Graph):
                transform = transform.copy()

            if isinstance(transform, cls):
                if transform._pd_group:
                    list_flat.append(transform)
                else:
                    list_flat += transform.transforms
            else:
                list_flat.append(transform)

        return list_flat

    def _gather_args_for_node(self, node: Node, inputs: dict):
        """Gather args to call node.transform

        This gathers all the args and arranges with according to the
        input_slice order.
        """
        gathered_args = [None for _ in self.parents[node]]

        for idx, parent_node in enumerate(self.parents[node]):
            output_slice, input_slice = self.edges[parent_node][node]
            arg = inputs[parent_node]
            if output_slice is not None:
                arg = arg[output_slice]
            # TODO: make sure there is always input_slice and non overlapping input_slice
            if input_slice is not None:
                gathered_args[input_slice] = arg
            else:
                gathered_args[idx] = arg

        if len(gathered_args) == 1:
            return gathered_args[0]
        return tuple(gathered_args)

    def connect(self, node_a, node_b, output_slice=None, input_slice=None):
        """Connect node_a with node_b

        node_a -> node_b
        """
        self.edges[node_a][node_b] = (output_slice, input_slice)
        self.parents[node_b].append(node_a)

    def connect_graph(self, node_a, graph_b, output_slice=None, input_slice=None, parent_position=None):
        """Connect node_a with a graph graph_b
        node_a -> input_node of graph_b -> ...

        :param node_a:
        :param graph_b:
        :param output_slice:
        :param input_slice:
        :param parent_position:
        :return:
        """

        for node_b in graph_b.edges[graph_b.input_node]:
            temp_output_slice = output_slice
            node_b_output_slice, input_slice = graph_b.edges[graph_b.input_node][node_b]
            if self._check_edge_compatibility(node_b_output_slice, input_slice, parent_position):
                temp_output_slice = node_b_output_slice if temp_output_slice is None else temp_output_slice
                temp_output_slice = None if temp_output_slice == parent_position else temp_output_slice
                self.connect(node_a,
                             node_b,
                             output_slice=temp_output_slice,
                             input_slice=input_slice,
                             )

        temp_edges = defaultdict(dict)
        temp_parents = defaultdict(list)

        for parent, children in graph_b.edges.items():
            if parent in (graph_b.input_node, graph_b.output_node):
                continue
            for child in children:
                if child not in (graph_b.input_node, graph_b.output_node):
                    temp_edges[parent][child] = graph_b.edges[parent][child]
        for parent, children in self.edges.items():
            for child in children:
                temp_edges[parent][child] = self.edges[parent][child]

        for child, parents in graph_b.parents.items():
            if child in (graph_b.input_node, graph_b.output_node):
                continue
            for parent in parents:
                if parent in (graph_b.input_node, graph_b.output_node):
                    continue
                temp_parents[child].append(parent)
        for child, parents in self.parents.items():
            for parent in parents:
                temp_parents[child].append(parent)

        self.edges = temp_edges
        self.parents = temp_parents

    @lru_cache()
    def sorted_nodes(self):
        """Topologically sorted nodes"""
        return _topological_node_sort(self.input_node, self.edges, self.parents)

    def __call__(self, args):
        """Call Method for Graph

        Apply Breadth-First transforms

        In __ C __ X __ Out
          \__ D __ Y __/

        In > C > D > X > Out
        """
        results = {self.input_node: args}
        for node in self.sorted_nodes()[1:]:
            node_arg = self._gather_args_for_node(node, results)
            results[node] = node.call_node(node_arg)
            # TODO: remove results that aren't needed any more

        return results[self.output_node]

    def convert_to_networkx(self):
        networkx_graph = nx.DiGraph()
        for parent, children_dict in self.edges.items():
            networkx_graph.add_node(parent.name_id, node=parent)
            for child in children_dict:
                networkx_graph.add_node(child.name_id, node=child)
                networkx_graph.add_edge(parent.name_id, child.name_id)
        self.networkx_graph = networkx_graph

    def draw(self):
        """Draw the graph

        :return:
        """
        self.convert_to_networkx()
        dot = nx.nx_agraph.to_agraph(self.networkx_graph)
        dot.layout('dot')
        return Image(dot.draw(format='png', prog='dot'))

    @staticmethod
    def _check_edge_compatibility(output_slice, input_slice, parent_pos):
        """Check if given edge detail gives a valid path

        :param output_slice:
        :param input_slice:
        :param parent_pos:
        :return:
        """
        # TODO: Check, TEST & Simplify
        input_slice = input_slice if input_slice is not None else parent_pos
        if input_slice is None:
            return True
        if isinstance(input_slice, slice):
            if input_slice.stop is None:
                return True
            input_slice = range(input_slice.stop)[input_slice]
        if output_slice is None:
            return True
        elif isinstance(output_slice, slice):
            if output_slice.stop is None:
                return True
            output_slice = range(output_slice.stop)[output_slice]
        if isinstance(input_slice, int) and isinstance(output_slice, list):
            if input_slice in output_slice:
                return True
            return False
        elif isinstance(input_slice, list) and isinstance(output_slice, int):
            if output_slice in input_slice:
                return True
            return False
        elif isinstance(input_slice, int) and isinstance(output_slice, int):
            if input_slice == output_slice:
                return True
            return False
        else:
            if len(set(input_slice).intersection(set(input_slice))) > 0:
                return True
            return False

    def list_all_paths(self, inp_node=None, path=None):
        """List all paths

        :param inp_node:
        :param path:
        """
        if path is None:
            path = []
        if inp_node is None:
            path = [self.input_node]
            inp_node = self.input_node

        if len(self.edges[inp_node]) == 0:
            return path

        return_path = []

        for idx, out_node in enumerate(self.edges[inp_node]):
            parent_position = self.parents[out_node].index(inp_node)

            output_slice, input_slice = self.edges[inp_node][out_node]
            if not self._check_edge_compatibility(output_slice, input_slice, parent_position):
                continue
            path_copy = path.copy()
            if isinstance(out_node.transform, Graph):
                path_copy.append(out_node.transform.input_node)
                return_path.append(out_node.transform.list_all_paths(out_node.transform.input_node, path_copy))
            else:
                path_copy.append(out_node)
                return_path.append(self.list_all_paths(out_node, path_copy))
        if len(return_path) == 1:
            return return_path[0]
        return return_path

    def _generate_preprocess_dict(self,
                                  current_node,
                                  preprocess_edges=None,
                                  batchify_node=None,
                                  batchify_found=False):
        """

        :param current_node:
        :param preprocess_edges:
        :param batchify_node:
        :param batchify_found:
        :return:
        """
        if batchify_node is None:
            batchify_node = dict()
        if preprocess_edges is None:
            preprocess_edges = defaultdict(dict)

        for child in self.edges[current_node]:
            if isinstance(child.transform, Batchify):
                batchify_found = True
                if child not in batchify_node:
                    batchify_node[child] = Node(identity - 'batchify Marker')
                preprocess_edges[current_node][batchify_node[child]] = self.edges[current_node][child]
                preprocess_edges[batchify_node[child]][self.output_node] = (None, None)
            elif child == self.output_node and batchify_found:
                raise SyntaxError('If a path has batchify, all paths must contain batchify')
            else:
                preprocess_edges[current_node][child] = self.edges[current_node][child]
                preprocess_edges, batchify_found = self._generate_preprocess_dict(child,
                                                                                  preprocess_edges,
                                                                                  batchify_node,
                                                                                  batchify_found)
        return preprocess_edges, batchify_found

    def _generate_forward_dict(self,
                               current_node=None,
                               forward_edges=None,
                               unbatchify_node=None,
                               unbatchify_found=False,
                               ):
        if unbatchify_node is None:
            unbatchify_node = dict()
        if forward_edges is None:
            forward_edges = defaultdict(dict)
        if current_node is None:
            batchify_nodes = [n_ for n_ in self.edges if isinstance(n_.transform, Batchify)]
            for batch_node in batchify_nodes:
                forward_edges[self.input_node][batch_node] = (None, None)
                forward_edges, unbatchify_found = self._generate_forward_dict(batch_node, forward_edges,
                                                                              unbatchify_node,
                                                                              unbatchify_found)
            return forward_edges, unbatchify_found

        for child in self.edges[current_node]:
            if isinstance(child.transform, Batchify):
                raise SyntaxError("Two batchify in same path is not allowed")
            elif isinstance(child.transform, Unbatchify):
                unbatchify_found = True
                if child not in unbatchify_node:
                    unbatchify_node[child] = Node(identity - 'unbatch Marker')
                forward_edges[current_node][unbatchify_node[child]] = self.edges[current_node][child]
                forward_edges[unbatchify_node[child]][self.output_node] = (None, None)
            elif child == self.output_node and unbatchify_found:
                raise SyntaxError('If a path has unbatchify, all paths must contain unbatchify')
            else:
                forward_edges[current_node][child] = self.edges[current_node][child]
                forward_edges, unbatchify_found = self._generate_forward_dict(child, forward_edges, unbatchify_node,
                                                                              unbatchify_found=unbatchify_found)
        return forward_edges, unbatchify_found

    def _generate_postprocess_dict(self,
                                   current_node=None,
                                   postprocess_edges=None,
                                   ):
        if postprocess_edges is None:
            postprocess_edges = defaultdict(dict)

        if current_node is None:
            unbatchify_nodes = [n_ for n_ in self.edges if isinstance(n_.transform, Unbatchify)]
            for unbatch_node in unbatchify_nodes:
                postprocess_edges[self.input_node][unbatch_node] = (None, None)
                postprocess_edges = self._generate_postprocess_dict(unbatch_node, postprocess_edges)
            return postprocess_edges

        for child in self.edges[current_node]:
            if isinstance(child.transform, Unbatchify):
                raise SyntaxError("Two Unbatchify in same path is not allowed")
            else:
                postprocess_edges[current_node][child] = self.edges[current_node][child]
                postprocess_edges = self._generate_postprocess_dict(child, postprocess_edges)
        return postprocess_edges

    def _pd_splits(self, input_components=0):
        """Generate splits

        :param input_components:
        :return:
        """
        preprocess_edges_dict, batchify_found = self._generate_preprocess_dict(self.input_node)
        forward_edges_dict, unbatchify_found = self._generate_forward_dict()
        postprocess_edges_dict = self._generate_postprocess_dict()

        build_splits = []
        if batchify_found:
            self._preprocess_edges = preprocess_edges_dict
            self._forward_edges = forward_edges_dict
            build_splits += ['preprocess', 'forward']
        else:
            self._pd_preprocess = Compose([identity - 'Preprocess Identity'])
            self._preprocess_edges = self._pd_preprocess.edges
            self._preprocess_parents = self._pd_preprocess.parents

            self._forward_edges = preprocess_edges_dict
            build_splits += ['forward']

        if len(postprocess_edges_dict) > 0:
            self._postprocess_edges = postprocess_edges_dict
            build_splits += ['postprocess']
        else:
            self._pd_postprocess = Compose([identity - 'Postprocess Identity'])
            self._postprocess_edges = self._pd_postprocess.edges
            self._postprocess_parents = self._pd_postprocess.parents

        if 'preprocess' in build_splits:
            self._preprocess_parents = _generate_parents_dict_from_edge_dict(self._preprocess_edges)
            self._preprocess_list = _convert_to_structured_list(start_node=self.input_node,
                                                                edges_dict=self._preprocess_edges,
                                                                parents_dict=self._preprocess_parents,
                                                                exclude_start_node=True,
                                                                )
            self._pd_preprocess = _build_transform_from_list(self._preprocess_list)

        if 'forward' in build_splits:
            self._forward_parents = _generate_parents_dict_from_edge_dict(self._forward_edges)
            self._forward_list = _convert_to_structured_list(start_node=self.input_node,
                                                             edges_dict=self._forward_edges,
                                                             parents_dict=self._forward_parents,
                                                             exclude_start_node=True,
                                                             )
            self._pd_forward = _build_transform_from_list(self._forward_list)

        if 'postprocess' in build_splits:
            self._postprocess_parents = _generate_parents_dict_from_edge_dict(self._postprocess_edges)
            self._postprocess_list = _convert_to_structured_list(start_node=self.input_node,
                                                           edges_dict=self._postprocess_edges,
                                                           parents_dict=self._postprocess_parents,
                                                           exclude_start_node=True,
                                                           )
            self._pd_postprocess = _build_transform_from_list(self._postprocess_list)

        return None, (self._pd_preprocess, self._pd_forward, self._pd_postprocess), True


class Compose(Graph):
    op = '>>'
    _name = 'compose'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        # if transforms are duplicated make a copy.
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

        current_nodes = [self.input_node]

        for transform in self.transforms:
            if isinstance(transform, Graph):
                for idx, node_a in enumerate(current_nodes):
                    self.connect_graph(node_a,
                                       transform,
                                       parent_position=idx if len(current_nodes) > 1 else None,
                                       output_slice=None,
                                       input_slice=None)
                current_nodes = transform.parents[transform.output_node]
                continue

            next_node = Node(transform)
            for node_a in current_nodes:
                self.connect(node_a,
                             next_node,
                             output_slice=node_a.pd_output_slice,
                             input_slice=next_node.pd_input_slice)
            current_nodes = [next_node]
        for node_a in current_nodes:
            self.connect(node_a, self.output_node)


class Rollout(Graph):
    op = '+'
    _name = 'rollout'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

        for transform in self.transforms:
            if isinstance(transform, Graph):
                self.connect_graph(self.input_node, transform)
                out_nodes = transform.parents[transform.output_node]
                for out_node in out_nodes:
                    self.connect(out_node,
                                 self.output_node)
                continue
            node = Node(transform)
            self.connect(self.input_node,
                         node,
                         output_slice=self.input_node.pd_output_slice,
                         input_slice=node.pd_input_slice)
            self.connect(node,
                         self.output_node,
                         output_slice=node.pd_output_slice,
                         input_slice=self.output_node.pd_input_slice)


class Parallel(Graph):
    op = '/'
    _name = 'parallel'

    def __init__(self, transforms: Iterable[Transform], call_info: inspector.CallInfo = None,
                 pd_name: Optional[str] = None, pd_group: bool = False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)

        for idx, transform in enumerate(self.transforms):
            if isinstance(transform, Parallel):
                self.connect_graph(self.input_node,
                                   transform,
                                   output_slice=idx)
                out_nodes = transform.parents[transform.output_node]
                out_node = out_nodes[idx]
                self.connect(out_node,
                             self.output_node)
                continue
            if isinstance(transform, Graph):
                self.connect_graph(self.input_node,
                                   transform,
                                   output_slice=idx,
                                   parent_position=None)
                out_nodes = transform.parents[transform.output_node]
                for out_node in out_nodes:
                    self.connect(out_node,
                                 self.output_node)
                continue
            node = Node(transform)
            self.connect(self.input_node,
                         node,
                         output_slice=idx,
                         input_slice=node.pd_input_slice)
            self.connect(node,
                         self.output_node,
                         output_slice=node.pd_output_slice,
                         input_slice=self.output_node.pd_input_slice)


def _check_parallel(children):
    """Check if current dict passed is Parallel or not"""
    output_slices = []
    for child, (output_slice, input_slice) in children.items():
        output_slices.append(output_slice)
    try:
        return sorted(output_slices) == list(range(len(children)))
    except TypeError:
        return False


def _helper_convert_compose(edges_dict=None,
                            parents_dict=None,
                            current_node=None,
                            current_type=None,
                            transform_list=None,
                            meta_transform_list=None,
                            nodes_left=None,
                            input_node=None):
    """Helper function to convert compose to list"""
    if transform_list is None:
        transform_list = []
    if meta_transform_list is None:
        meta_transform_list = []
    children = edges_dict[current_node]
    child_node = list(children.keys())[0]

    # Multiple parents marks end of Compose
    continue_compose = len(parents_dict[child_node]) == 1

    if current_type == Compose and continue_compose:
        transform_list += list(children.keys())
        nodes_left.remove(transform_list[-1])
    else:
        current_type = Compose
        transform_list = [current_type, current_node] if current_node != input_node else [current_type]
        nodes_left.remove(current_node)
        if continue_compose:
            transform_list += list(children.keys())
            nodes_left.remove(transform_list[-1])

        meta_transform_list.append(transform_list)

    if continue_compose:
        current_node = transform_list[-1]
        transform_list, meta_transform_list, nodes_left = _helper_convert_to_operators(
            edges_dict,
            parents_dict,
            current_node,
            current_type,
            transform_list,
            meta_transform_list,
            nodes_left=nodes_left,
            input_node=input_node,
        )

    return transform_list, meta_transform_list, nodes_left


def _helper_convert_to_operators(edges_dict=None,
                                 parents_dict=None,
                                 current_node=None,
                                 current_type=None,
                                 transform_list=None,
                                 meta_transform_list=None,
                                 nodes_left=None,
                                 input_node=None,
                                 ):
    """Helper function to convert given edge_dict to operator list

    :param edges_dict:
    :param parents_dict:
    :param current_node:
    :param current_type:
    :param transform_list:
    :param meta_transform_list:
    :param nodes_left:
    :param input_node:
    :return:
    """
    if transform_list is None:
        transform_list = []
    if meta_transform_list is None:
        meta_transform_list = []
    children = edges_dict[current_node]
    if len(children) == 0:
        return transform_list, meta_transform_list, nodes_left

    if len(children) == 1:
        transform_list, meta_transform_list, nodes_left = _helper_convert_compose(
            edges_dict=edges_dict,
            parents_dict=parents_dict,
            current_node=current_node,
            current_type=current_type,
            transform_list=transform_list,
            meta_transform_list=meta_transform_list,
            nodes_left=nodes_left,
            input_node=input_node,
        )
        return transform_list, meta_transform_list, nodes_left

    if current_type is None and current_node != input_node:
        meta_transform_list.append(current_node)

    if _check_parallel(children):
        # PARALLEL
        transform_list = [Parallel]
        meta_transform_list.append(transform_list)
        for idx, child in enumerate(children):
            _, _, nodes_left = _helper_convert_to_operators(edges_dict=edges_dict,
                                                            parents_dict=parents_dict,
                                                            current_node=child,
                                                            current_type=Parallel,
                                                            transform_list=None,
                                                            meta_transform_list=transform_list,
                                                            nodes_left=nodes_left,
                                                            input_node=input_node,
                                                            )
        return transform_list, meta_transform_list, nodes_left

    # ROLLOUT
    transform_list = [Rollout]
    meta_transform_list.append(transform_list)
    for idx, child in enumerate(children):
        _, _, nodes_left = _helper_convert_to_operators(edges_dict=edges_dict,
                                                        parents_dict=parents_dict,
                                                        current_node=child,
                                                        current_type=Rollout,
                                                        transform_list=None,
                                                        meta_transform_list=transform_list,
                                                        nodes_left=nodes_left,
                                                        input_node=input_node,
                                                        )
    return transform_list, meta_transform_list, nodes_left


def _convert_to_structured_list(start_node, edges_dict, parents_dict, exclude_start_node=True):
    """Convert an edge_dict to list of operations

    :param start_node: node to start the conversion from
    :param edge_dict: dict of edges_dict of nodes
    :param parents_dict: dict of parents of nodes
    :param exclude_start_node: True if start_node is not to be included
    :return: list of operations
    """

    nodes_left = _topological_node_sort(start_node, edges_dict, parents_dict)

    meta_transform_list = []
    transform_list = []
    current_node = start_node

    if not exclude_start_node:
        start_node = None

    while len(nodes_left) > 2:
        transform_list, meta_transform_list, nodes_left = _helper_convert_to_operators(
            edges_dict=edges_dict,
            parents_dict=parents_dict,
            current_node=current_node,
            current_type=None,
            transform_list=transform_list,
            meta_transform_list=meta_transform_list,
            nodes_left=nodes_left,
            input_node=start_node,
        )
        current_node = nodes_left[0]

    return meta_transform_list


def _build_transform_from_list(input_list):
    """Build transforms from a list

    input_list can have following elements:
    1. transforms
    2. list starting with Class (Compose/Rollout/Parallel) and contains transforms
        e.g. [Compose, t1, t2, t3]

    :param input_list: list of transforms
    """
    current_type = input_list[0]
    start_pos = 1
    if not inspect.isclass(current_type):
        current_type = Compose
        start_pos = 0

    transform_list = []
    for l_ in input_list[start_pos:]:
        if isinstance(l_, list):
            transform_list.append(_build_transform_from_list(l_))
        else:
            transform_list.append(l_.transform)
    return current_type(transform_list)


def _generate_parents_dict_from_edge_dict(edge_dict):
    """Generate parents dict

    :param edge_dict: dict of edges_dict for nodes
    :return:
    """
    parents = defaultdict(list)
    for parent, children in edge_dict.items():
        for child in children:
            parents[child].append(parent)
    return parents


def _topological_node_sort(start_node, edges_dict, parents_dict):
    """Sort nodes topologically (BFS)

    :param start_node: Node to start the search with
    :param edges_dict: dict of edges_dict for nodes
    :param parents_dict: dict of parents for nodes
    :return: sorted list of nodes
    """
    queue = [start_node]
    sorted_ = []
    while queue:
        node = queue.pop(0)
        sorted_.append(node)
        queue += [child for child in edges_dict[node].keys() if all(p in sorted_ for p in parents_dict[child])]
    return sorted_
