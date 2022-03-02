"""The Transform class and its fundamental children.

Transforms should be created using the `padl.transform` wrap-function.
"""
import ast
import re
from copy import copy
from collections import Counter, namedtuple, OrderedDict
from functools import lru_cache
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
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

import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader

from padl.dumptools import symfinder, inspector
from padl.dumptools.var2mod import CodeGraph, CodeNode, find_codenode
from padl.dumptools.symfinder import ScopedName
from padl.dumptools.serialize import Serializer
from padl.dumptools.sourceget import replace
from padl.dumptools.ast_utils import get_position

from padl.dumptools.packagefinder import dump_packages_versions
from padl.exceptions import WrongDeviceError
from padl.print_utils import combine_multi_line_strings, create_reverse_arrow, make_bold, \
    make_green, create_arrow, format_argument, visible_len


_pd_trace = []
MAX_WIDTH_FOR_PRINT = 110


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
            # if the call was from here (i.e. transform.py), ignore the scope
            calling_module_name = inspector.non_init_caller_frameinfo().frame.f_globals['__name__']
            call_info = inspector.CallInfo(ignore_scope=calling_module_name == __name__)
        self._pd_call_info = call_info
        self._pd_varname = {}
        self._pd_name = pd_name
        self._pd_device = 'cpu'
        self._pd_traceback = traceback.extract_stack()
        self._pd_external_full_dump = False

    @property
    def _pd_full_dump_relevant_module(self):
        return self._pd_call_info.scope.module

    @property
    def _pd_full_dump(self) -> bool:
        """If *True*, dump the Transform in full (with definition etc) even if it was defined in
        a different module. Else, only dump an import statement. """
        module = self._pd_full_dump_relevant_module
        # always fully dump Transforms from the module the dump was triggered in
        if inspector.caller_module() == module:
            return True
        # always fully dump from __main__
        if getattr(module, '__name__', '__main__') == '__main__':
            return True
        # always fully dump loaded transforms
        if getattr(module, '_pd_full_dump', False):
            return True
        # fully dump all Transforms from packages or modules specified in
        # _pd_external_full_dump_modules
        if self._pd_is_full_dump_module(module.__spec__.name):
            return True
        return self._pd_external_full_dump

    def _pd_is_full_dump_module(self, module_name):
        return any(module_name.startswith(mod) for mod in self._pd_external_full_dump_modules)

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
        _, splits, has_batchify, has_unbatchify = self._pd_splits()
        if has_batchify and has_unbatchify:
            preprocess, forward, postprocess = splits
        elif has_batchify and not has_unbatchify:
            preprocess, forward, postprocess = splits[0], splits[1], identity
        else:
            # case when no batchify and no unbatchify
            preprocess, forward, postprocess = identity, splits[0], identity
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
            False,
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
        named_copy._pd_varname = {}
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
        opt = None
        if hasattr(self, 'pd_save_options'):
            opt = self._pd_process_options(self.pd_save_options)
        # pd_pre_save requires default behaviour on receiving None
        try:
            if 'options' in inspect.signature(self.pre_save).parameters:
                return self.pre_save(path, i, options=opt)
            return self.pre_save(path, i)
        except AttributeError:
            pass

    def pd_post_load(self, path: Path, i: int, options: Optional[dict] = None):
        """Method that is called on each transform after loading.

        This normally does nothing. Override to implement custom serialization.

        :param path: The load path.
        :param i: Unique transform index, can be used to construct filenames.
        :param options: Options dictionary (optional) to control saving behaviour, comes
                        from attribute "self.pd_save_options"
        """
        opt = None
        if options is not None:
            opt = self._pd_process_options(options)
        try:
            if 'options' in inspect.signature(self.pre_save).parameters:
                return self.post_load(path, i, options=opt)
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

    @classmethod
    def _pd_process_options(cls, options):
        """
        This helper finds a key in options which is either cls or is an
        ancestor of cls, and is unique and minimal with respect to the ancestral
        relation.
        """
        try:
            return options[cls]
        except KeyError:
            pass
        mro_check = sorted(
            [(len(x.mro()), x) for x in options.keys() if issubclass(cls, x)],
            key=lambda x: x[0]
        )
        if not mro_check:
            return
        # print not sure if this case would ever arise due to
        # inbuilt error "Cannot create a consistent method resolution order (MRO) for bases"
        if mro_check[1:] and mro_check[0][0] == mro_check[1][0]:
            raise Exception(f'Couldn\'t find a clear option applying to {cls}'
                            f'Found at least two candidates: {mro_check[0][1]} '
                            f'and {mro_check[1][1]}')
        return options[mro_check[0][1]]

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
        start = CodeNode.from_source(start_source, scope, name=name or "_pd_dummy")
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
            next_codenode = find_codenode(next_name, self._pd_external_full_dump_modules)

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
                  path: Optional[Path] = None,
                  options: Optional[dict] = None) -> Union[str, Tuple[str, str]]:
        """Dump the transform as python code.

        :param return_versions: If *True* return a tuple of the code and a file listing
            dependencies and their versions.
        :param path: Optional path to save at, might be required for serializer code snippets.
        """
        graph = self._pd_build_codegraph(name='_pd_main')
        if options is not None:
            graph.append
        Serializer.save_all(graph, path)
        code = graph.dumps()
        if return_versions:
            versions = dump_packages_versions(node.ast_node for node in graph.values())
            return code, versions
        return code

    def __repr__(self):
        return self._pd_shortrepr(formatting=False)

    def _repr_pretty_(self, p, cycle):
        # pylint: disable=invalid-name
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

    def _pd_parensrepr(self, formatting=True, max_width=None) -> str:
        short = self._pd_shortrepr(formatting, max_width=max_width)
        max_width = 50 if max_width is None else max_width
        if len(short) < max_width:
            if len(getattr(self, 'transforms', [])) > 1:
                short = f"{make_green('[', not formatting)}{short}{make_green(']', not formatting)}"
            return short
        return self._pd_tinyrepr(formatting)

    def _pd_shortrepr(self, formatting=True, max_width=None) -> str:
        # pylint: disable=unused-argument
        """A short string representation of the transform.
        :param max_width: maximum width for the string representation
        """
        return self._pd_title(max_width=max_width)

    def _pd_tinyrepr(self, formatting=True) -> str:
        """A tiny string representation of the transform.
        """
        return self.pd_name or f'<anonymous {self.__class__.__name__}>'

    def _pd_title(self, max_width=None) -> str:
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
        """Check if all Transforms in the "forward" part are on the correct device.

        All transforms in the "forward" part of a Pipeline need to be on the same device as
        specified for the whole Pipeline.
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
        """Return *True* if to arguments should be unpacked, else *False*."""
        signature_count = 0
        if not isinstance(arg, (list, tuple)):
            return False

        if getattr(self, '_pd_number_of_inputs', None) is not None:
            return self._pd_number_of_inputs > 1  # pylint: disable=no-member

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

        :param arg: Argument to call the transform with.
        :param mode: The mode ("infer", "eval", "train") to perform the call with.
        :param ignore_grad: If *True* gradient settings are ignored.
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
            grad_before = torch.is_grad_enabled()  # pylint: disable=no-member
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
        # pylint: disable=no-self-use
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
                        self._pd_trace_error(None, [args[i] for i in ix])
                        raise err

                if use_post or flatten:
                    output = _unpack_batch(output)
                    if use_post:
                        try:
                            output = [post.pd_call_in_mode(x, mode, ignore_grad=True)
                                      for x in output]
                        except Exception as err:
                            self._pd_trace_error(None, [args[i] for i in ix])
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
        """Return the device ("cpu" / "cuda") the Transform is on."""
        return self._pd_device

    @pd_device.setter
    def pd_device(self, device: str):
        self._pd_device = device

    @property
    def pd_preprocess(self) -> "Transform":
        """The preprocessing part of the Transform.

        The device must be propagated from self."""
        pre = self.pd_stages[0]
        pre.pd_to(self.pd_device)
        return pre

    @property
    def pd_forward(self) -> "Transform":
        """The forward part of the Transform (that what's typically done on the GPU).

        The device must be propagated from self."""
        forward = self.pd_stages[1]
        forward.pd_to(self.pd_device)
        return forward

    @property
    def pd_postprocess(self) -> "Transform":
        """The postprocessing part of the Transform.

        The device must be propagated from self."""
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
        """Get a list with all pytorch layers in the Transform (including layers in sub-transforms).
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

    def pd_get_loader(self, args, preprocess: 'Transform', mode: str, **kwargs) -> DataLoader:
        """Get a pytorch data loader applying *preprocess* to *args*.

        :param args: A sequence of datapoints.
        :param preprocess: Preprocessing Transform.
        :param mode: PADL mode to call the preprocess Transform in.
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
            **kwargs
        )

    def infer_apply(self, inputs=()):
        """Call the Transform within the infer context.

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
                self._pd_trace_error(None, in_args)
                raise err
        if use_post:
            try:
                inputs = postprocess.pd_call_in_mode(inputs, mode='infer', ignore_grad=True)
            except Exception as err:
                self._pd_trace_error(None, in_args)
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

    def _pd_title(self, max_width=None) -> str:
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
    :param wrap_type: One of {'module', 'lambda', 'decorator', 'inline'} - specifying how the was
        function was wrapped.
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
        if (self._pd_full_dump
                or self._wrap_type in ('module', 'lambda')
                or (self._wrap_type != 'inline' and self._pd_call_info.scope.scopelist)):
            return super()._pd_codegraph_add_startnodes(graph, name)
        module = inspector.caller_module()
        scope = symfinder.Scope.toplevel(module)
        source = f'from {self.__module__} import {self.__name__}'

        if self._wrap_type == 'inline':
            node = CodeNode.from_source(source, scope, name=self.__name__)
            graph[ScopedName(self.__name__, scope, 0)] = node
            emptyscope = symfinder.Scope.empty()
            graph[ScopedName('transform', emptyscope, 0)] = \
                CodeNode.from_source('from padl import transform', emptyscope, name='transform')

            start_source = f'{name or "_pd_dummy"} = transform({self.__name__})'
            start = CodeNode.from_source(start_source, scope, name=name or "_pd_dummy")
            if name is not None:
                graph[ScopedName(name, scope, 0)] = start
            return {}

        if name is not None:
            source += f' as {name}'
        else:
            name = self.__name__
        node = CodeNode.from_source(source, scope, name=name)
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
        """The function's signature. """
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

    def _pd_shortrepr(self, formatting=True, max_width=None) -> str:
        if len(self._pd_longrepr().split('\n', 1)) == 1:
            return self._pd_longrepr(formatting)
        return super()._pd_shortrepr(formatting)

    def _pd_title(self, max_width=None) -> str:
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

    def __name__(self):
        """(For completeness. A FunctionTransform gets its :meth:`__name__` when it's being wrapped
        (see :func:`~padl.wrap._wrap_function`)"""
        raise NotImplementedError()


class ClassTransform(AtomicTransform):
    """Class Transform.

    Do not use this directly, instead, use the `transform` decorator to wrap a class.

    :param pd_name: name of the transform
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    :param arguments: ordered dictionary of initialization arguments to be used in printing
    """

    def __init_subclass__(cls, *_args, **_kwargs):
        cls._pd_class_call_info = inspector.CallInfo()

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

    def _pd_evaluable_repr_inner(self, indent: int = 0) -> str:
        if not self._pd_full_dump and self._pd_call_info.scope.module_name == self.__class__.__module__:
            repr_ = self.pd_varname(self._pd_call_info.scope)
            if repr_ is not None:
                return repr_
        return super()._pd_evaluable_repr_inner(indent)

    @property
    def _pd_full_dump_relevant_module(self):
        return inspect.getmodule(self.__class__)

    def _pd_codegraph_add_startnodes_import_var(self, graph, name):
        instance_scope = self._pd_call_info.scope
        varname = self.pd_varname(instance_scope)

        import_source = f'from {self.__module__} import {varname}'
        import_node = CodeNode.from_source(import_source, instance_scope, name=varname)

        graph[ScopedName(varname, instance_scope, 0)] = import_node

        if name != varname:
            start_source = f'{name or "_pd_dummy"} = {varname}'
            start_node = CodeNode.from_source(start_source,
                                              instance_scope,
                                              name=name or "_pd_dummy")
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
        import_node = CodeNode.from_source(import_source, instance_scope,
                                           name=self.__class__.__name__)
        graph[ScopedName(self.__class__.__name__, instance_scope, 0)] = import_node
        nodes = [import_node]

        # make the call
        call = self.__class__.__name__ + f'({self._pd_split_call()[1]})'
        call_scope = symfinder.Scope.toplevel(inspector.caller_module())
        start_source = f'{name or "_pd_dummy"} = {call}'
        start_node = CodeNode.from_source(start_source, instance_scope, name=name or "_pd_dummy")
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

        call = self.__class__.__name__ + f'({self._pd_split_call()[1]})'
        start_source = f'{name or "_pd_dummy"} = {call}'
        start_node = CodeNode.from_source(start_source, call_scope, name=name or '_pd_dummy')
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
        (body_msg, _), _, _ = symfinder.find_in_scope(ScopedName(self.__class__.__name__,
                                                                 self._pd_call_info.scope))
        try:
            return 'class ' + body_msg.split('class ', 1)[1]
        except IndexError:
            return body_msg

    def _pd_split_call(self):
        """Split class initialization call from its arguments.

        :return: A tuple of class name and arguments.
        """
        return symfinder.split_call(self._pd_call)

    def _formatted_args(self, max_width=None) -> str:
        """Format the object's init arguments for printing. """
        if self._pd_arguments is None:  # fall back
            return self._pd_split_call()[1]

        args_list = []
        for key, value in self._pd_arguments.items():
            if key == 'args':
                args_list += [f'{format_argument(val)}' for val in value]
            elif key == 'kwargs':
                args_list += [f'{subkey}={format_argument(val)}' for subkey, val in value.items()]
            else:
                args_list.append(f'{key}={format_argument(value)}')

        if max_width is None:
            return ', '.join(args_list)

        max_args_list = []
        max_width -= 3
        for args in args_list:
            max_args_list += [args]
            if visible_len(', '.join(max_args_list)) > max_width:
                max_args_list.pop(-1)
                max_args_list.append('...')
                break
        return ', '.join(max_args_list)

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        try:
            str_ = self.source.split('\n')[:30]
            if marker:
                return str_[0] + marker[1] + '\n' + '\n'.join(str_[1:])
            return '\n'.join(str_)
        except symfinder.NameNotFound:
            return self._pd_call + marker[1] if marker else self._pd_call

    def _pd_title(self, max_width=None) -> str:
        title = type(self).__name__
        if max_width is not None:
            max_width -= visible_len(title)
        return title + '(' + self._formatted_args(max_width=max_width) + ')'


class TorchModuleTransform(ClassTransform):
    # pylint: disable=no-member
    """Transform class for use with `torch.nn.Module`."""

    @property
    @lru_cache(maxsize=128)
    def _pd_signature(self):
        return inspect.signature(self.forward).parameters

    def pre_save(self, path: Path, i: int, options: Optional[Any] = None):
        """Dump the model's parameters to a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        if isinstance(options, str) and options == 'no-save':
            return
        path = Path(path)
        checkpoint_path = path / f'{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def post_load(self, path, i, options: Optional[Any] = None):
        """Load the model's parameters form a save-folder.

        :param path: The save-folder path.
        :param i: Unique transform index, used to construct filenames.
        """
        if isinstance(options, str) and options == 'no-save':
            return
        path = Path(path)
        checkpoint_path = path / f'{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
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

        self.transform: Transform = transform

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
            output_components, splits, has_batchify, has_unbatchify = \
                self.transform._pd_splits(input_components)
            return (
                # output_components is whatever the sub-transform does to it
                output_components,
                # the splits are the splits of the sub-transform, but mapped
                tuple(Map(split) if not isinstance(split, Identity) else identity
                      for split in splits),
                has_batchify,
                has_unbatchify
            )
        assert isinstance(input_components, list)

        # if it's a list, this means the input is structured and can potentially be in
        # different states (fresh, batchified, unbatchified)
        splits = ([], [], [])
        output_components = []
        has_batchify = False
        has_unbatchify = False

        for input_component in input_components:
            # for each input, we compute the *output_components* and the *splits* ...
            sub_output_components, sub_splits, sub_has_batchify, sub_has_unbatchify = \
                self.transform._pd_splits(input_component)
            has_batchify = has_batchify or sub_has_batchify
            has_unbatchify = has_unbatchify or sub_has_unbatchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        # .. and combine them as a Parallel
        return (
            output_components,
            tuple(Parallel(s) if s else identity for s in splits),
            has_batchify,
            has_unbatchify
        )

    def __call__(self, args: Iterable):
        """
        :param args: Args list to call transforms with
        """
        return tuple([self.transform._pd_unpack_args_and_call(arg) for arg in args])

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        inner = self.transform._pd_shortrepr(formatting)
        if isinstance(self.transform, Pipeline):
            inner = f'( {inner} )'
        str_ = '~ ' + inner
        return str_ + marker[1] if marker else str_

    def _pd_shortrepr(self, formatting=True, max_width=None):
        return self._pd_longrepr(formatting, marker=None)[:max_width]

    def _pd_tinyrepr(self, formatting=True):
        return f'~ {self.transform._pd_tinyrepr(formatting)}'

    def _pd_title(self, max_width=None):
        return f'Map of {self.transform._pd_title(max_width)}'

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
            for transform_ in self._pd_all_transforms():
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
            and not self._pd_call_info.scope.scopelist
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
        node = CodeNode.from_source(source, scope, name=name)
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
            graph[ScopedName('padl.group', emptyscope, 0)] = CodeNode.from_source('import padl',
                                                                                  emptyscope,
                                                                                  name='padl')

        # iterate over sub-transforms and update the codegraph with their codegraphs
        for transform in self.transforms:
            varname = transform.pd_varname(self._pd_call_info.scope)
            # pylint: disable=protected-access
            transform._pd_build_codegraph(graph, varname)
        return graph

    def _pd_longrepr(self, formatting=True, marker=None):
        between = f'\n{make_green(self.display_op, not formatting)}  \n'
        rows = [make_bold(f'{i}: ', not formatting) + t._pd_shortrepr(formatting)
                for i, t in enumerate(self.transforms)]
        return between.join(rows) + '\n'

    def _pd_shortrepr(self, formatting=True, max_width=None):
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

    def _pd_title(self, max_width=None):
        if max_width is not None:
            return self.__class__.__name__[:max_width]
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

    def _add_name_to_splits(self, final_splits):
        """Add name to split-transforms. """
        if self._pd_name is not None:
            for i, split in enumerate(final_splits):
                if not isinstance(split, Identity):
                    final_splits[i] = split - self._pd_name


class Compose(Pipeline):
    """Apply series of transforms on input.

    Compose([t1, t2, t3])(x) = t3(t2(t1(x)))

    :param transforms: List of transforms to compose.
    :param call_info: A `CallInfo` object containing information about the how the transform was
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

    def _pd_splits(self, input_components=0) -> Tuple[Union[int, List],
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
        has_unbatchify = False

        # for each sub-transform ...
        for transform_ in self.transforms:
            # ... see what comes out ...
            output_components, sub_splits, sub_has_batchify, sub_has_unbatchify = \
                transform_._pd_splits(output_components)

            has_batchify = has_batchify or sub_has_batchify
            has_unbatchify = has_unbatchify or sub_has_unbatchify
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
        return output_components, final_splits, has_batchify, has_unbatchify

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
        # Get maximum number of children in a single row
        max_children_number = max([
            sum([1 for s in t.transforms]) if hasattr(t, 'transforms')
            else 1
            for t in self.transforms
        ])
        max_width = int(MAX_WIDTH_FOR_PRINT/max_children_number)
        # pad the components of rows which are shorter than other parts in same column
        rows = [
            [s._pd_parensrepr(max_width=max_width) for s in t.transforms] if hasattr(t, 'transforms')
            else [t._pd_shortrepr(max_width=max_width)]
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
        """Call method for Compose.

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
        has_unbatchify = False

        for transform_ in self.transforms:
            # pylint: disable=protected-access
            sub_output_components, sub_splits, sub_has_batchify, sub_has_unbatchify = \
                transform_._pd_splits(input_components)
            has_batchify = has_batchify or sub_has_batchify
            has_unbatchify = has_unbatchify or sub_has_unbatchify

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
        return output_components, final_splits, has_batchify, has_unbatchify

    def __call__(self, args):
        """Call method for Rollout.

        :param args: Argument to call with.
        :return: `namedtuple` of outputs.
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
        between = f'\n{make_green_("│ " + self.display_op)}  \n'
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
        has_unbatchify = False

        for transform_, input_component in zip(self.transforms, input_components_):
            # and compute the sub-splits
            sub_output_components, sub_splits, sub_has_batchify, sub_has_unbatchify = \
                transform_._pd_splits(input_component)
            has_batchify = has_batchify or sub_has_batchify
            has_unbatchify = has_unbatchify or sub_has_unbatchify

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
        return output_components, final_splits, has_batchify, has_unbatchify

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
                out += f'{make_green_(pipes(len_ - i - 1) + spaces(i + 2) + self.display_op)}  \n'

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
        # ensure that all inputs are batchified.
        assert self._pd_merge_components(input_components) < 2, \
            'double unbatchify'
        # put the output component to 2 ("un-batchified")
        return 2, (identity, identity, self), False, True

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
        if isinstance(args, dict):
            return {k: self(args[k]) for k in args}

        raise TypeError('only tensors, dictionary and tuples of tensors recursively supported...')


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
        return 1, (self, identity, identity), True, False

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


def load(path, **kwargs):
    """Load a transform (as saved with padl.save) from *path*.

    Use keyword arguments to override params (see :func:`padl.param`).
    """
    if Path(path).is_file():
        return _zip_load(path)
    path = Path(path)
    with open(path / 'transform.py') as f:
        source = f.read()

    class _EmptyLoader(Loader):
        def create_module(self, spec):
            return types.ModuleType(spec.name)

    module_name = str(path).replace('/', '.').lstrip('.') + 'transform'
    spec = ModuleSpec(module_name, _EmptyLoader())
    module = module_from_spec(spec)

    pd_found_params = {}
    module.__dict__.update({
        '_pd_is_padl_file': True,
        '_pd_source': source,
        '_pd_module': module,
        '_pd_full_dump': True,
        '_pd_params': kwargs,
        '_pd_found_params': pd_found_params,
        '__file__': str(path / 'transform.py')
    })

    code = compile(source, path / 'transform.py', 'exec')

    # pylint: disable=exec-used
    exec(code, module.__dict__)

    for k in kwargs:
        if k not in pd_found_params:
            msg = (
                f'Parameter {k} does not exist.\n\n' +
                'Available parameters:\n' +
                '\n'.join(f'  {n} (default: {d})' if d is not None
                          else 'f'
                          for n, d in pd_found_params.items())
            )
            raise ValueError(msg)

    # pylint: disable=no-member,protected-access

    transform = module._pd_main
    for i, subtrans in enumerate(transform._pd_all_transforms()):
        if hasattr(transform, 'pd_save_options'):
            subtrans.pd_post_load(path, i, options=transform.pd_save_options)
        else:
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
    (0, 2)
    >>> ig[1]
    (1, 3)

    :param samples: An object implementing __getitem__ and __len__.
    :param transform: Preprocessing transform.
    :param entire_transform: :class:`Transform` which *transform* belongs to, i.e.,
        :class:`Transform` whose preprocessing part is *transform*.
    """

    def __init__(self, samples, transform, entire_transform=None):
        self.samples = samples
        self.transform = transform
        if entire_transform is None:
            self.entire_transform = self.transform
        else:
            self.entire_transform = entire_transform

    def __getitem__(self, item):
        try:
            return item, self.transform(self.samples[item])
        except Exception as err:
            self.entire_transform._pd_trace_error(None, [self.samples[item]])
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
