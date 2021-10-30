"""The Transform class and some of its children. """
import ast
import re
from copy import copy
from collections import Counter, namedtuple, OrderedDict
import contextlib
import inspect
from itertools import chain
from pathlib import Path
import types
from typing import Iterable, List, Literal, Optional, Set, Tuple, Union, Iterator
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from padl.data import SimpleIterator
from padl.dumptools import var2mod, thingfinder, inspector
from padl.dumptools.serialize import Serializer

from padl.dumptools.packagefinder import dump_packages_versions
from padl.exceptions import WrongDeviceError
from padl.print_utils import combine_multi_line_strings, create_reverse_arrow, make_bold, \
    make_green, create_arrow, format_argument


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
        if self._pd_name is None:
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
        return named_copy

    def __invert__(self) -> "Map":
        """Map.

        Example:
            t = ~a
        """
        return Map(self)

    @property
    def n_display_inputs(self):
        return len(self.pd_get_signature())

    @property
    def n_display_outputs(self):
        return 1

    @property
    def display_width(self):
        return len(self._pd_shortname())

    @property
    def children_widths(self):
        return [self.display_width]

    def pd_pre_save(self, path: Path, i: int):
        """Method that is called on each transform before saving.

        :param path: The save path.
        :param i: Subtransform index.
        """

    def pd_post_load(self, path: Path, i: int):
        """Method that is called on each transform after loading.

        :param path: The load path.
        :param i: Subtransform index.
        """

    def pd_save(self, path: Union[Path, str]):
        """Save the transform to *path*. """
        path = Path(path)
        path.mkdir(exist_ok=True)
        for i, subtrans in enumerate(self.pd_all_transforms()):
            subtrans.pd_pre_save(path, i)
        code, versions = self.pd_dumps(True, path=path)

        with open(path / 'transform.py', 'w') as f:
            f.write(code)
        with open(path / 'versions.txt', 'w') as f:
            f.write(versions)

    def _pd_codegraph_startnode(self, name: str) -> var2mod.CodeNode:
        """Build the start-code-node - the node with the source needed to create *self* as "name".
        (in the scope where *self* was originally created). """
        start_source = f'{name or "_pd_dummy"} = {self.pd_evaluable_repr()}'
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
                            scope: Optional[thingfinder.Scope] = None) -> Tuple[dict, dict]:
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        if scope is None:
            scope = self._pd_call_info.scope

        if name is not None:
            name_scope_here = name, scope
        else:
            name_scope_here = None

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

            if next_var.startswith('TADL_VALUE'):
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
                print(next_var, 'can deal with itself')
                continue

            # find how next_var came into being
            (source, node), scope_of_next_var = thingfinder.find_in_scope(next_var, next_scope)
            scopemap[next_var, next_scope] = scope_of_next_var

            # find dependencies
            globals_ = {
                (var, scope_of_next_var)  # TODO: this
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

    def pd_evaluable_repr(self, indent: int = 0, var_transforms: Optional[dict] = None) -> str:  # TODO: var_transforms needed?
        # pylint: disable=unused-argument,no-self-use
        """Return a string that if evaluated *in the same scope where the transform was created*
        creates the transform. """
        raise NotImplementedError

    def pd_all_transforms(self, result=None):
        """Return a list of all transforms needed for executing the transform.

        This includes the transform itself, the subtransforms of a compount transform or
        transforms a function-transform depends on as a global. """
        if result is None:
            result = []
        if self in result:
            return result
        result.append(self)
        for transform in self.pd_direct_subtransforms:
            transform.pd_all_transforms(result)
        return result

    @property
    def pd_direct_subtransforms(self):
        """Iterate over the direct subtransforms of this. """
        # pylint: disable=no-self-use
        raise NotImplementedError

    def pd_dumps(self, return_versions: bool = False, path=None) -> str:
        """Dump the transform as python code. """
        scope = thingfinder.Scope.toplevel(inspector.caller_module())
        graph, scopemap = self._pd_build_codegraph(name='_pd_main', scope=scope)
        Serializer.save_all(graph, scopemap, path)
        unscoped = var2mod.unscope_graph(graph, scopemap)
        code = var2mod.dumps_graph(unscoped)
        if return_versions:
            versions = dump_packages_versions(node.ast_node for node in graph.values())
            return code, versions
        return code

    def _pd_shortname(self, bold_title=False):
        title = self._pd_title()
        if bold_title:
            title = make_bold(title)
        if self.pd_name is not None and self.pd_name != title:
            return title + f' - "{self.pd_name}"'
        return title

    def _pd_title(self):
        raise NotImplementedError

    @staticmethod
    def _add_parentheses_if_needed(name):
        return name

    @staticmethod
    def _pd_add_format_to_str(name):
        """
        Create formatted output based on "name" input lines.

        :param name: line or lines of input
        """
        res = '    ' + '\n    '.join(name.split('\n')) + '\n'
        return res

    def _pd_bodystr(self):
        raise NotImplementedError

    def pd_repr(self, indent: int = 0) -> str:
        # pylint: disable=unused-argument
        varname = self.pd_varname()
        evaluable_repr = self.pd_evaluable_repr()
        if varname is None or varname == evaluable_repr:
            return f'{evaluable_repr}'
        return f'{evaluable_repr} [{varname}]'

    def __repr__(self) -> str:
        top_message = Transform._pd_shortname(self, True) + ':' + '\n\n'
        bottom_message = self._pd_bodystr()
        return top_message + self._pd_add_format_to_str(bottom_message)

    def _pd_find_varname(self, scopedict: dict) -> Optional[str]:
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

    def pd_varname(self) -> Optional[str]:
        """The name of the variable name the transform was last assigned to.

        Example:

        >>> foo = MyTransform()
        >>> foo._pd_varname
        "foo"

        :returns: A string with the variable name or *None* if the transform has not been assigned
            to any variable.
        """
        if self._pd_varname is _notset:
            self._pd_varname = self._pd_find_varname(self._pd_call_info.module.__dict__)
        return self._pd_varname

    def _pd_set_varname(self, val):  # TODO: needed (used in wrap, but can be done without "set" method potentially)
        self._pd_varname = val

    def _pd_forward_device_check(self):
        """Check all transform in forward are in correct device

        All transforms in forward need to be in same device as specified for
        the whole CompoundTransform.

        :return: Bool
        """
        for layer in self.pd_forward.pd_layers:
            for parameters in layer.parameters():
                if parameters.device.type != self.pd_device:
                    raise WrongDeviceError(self, layer)
        return True

    def _pd_unpack_argument(self, arg):
        """Returns True if to unpack argument else False"""
        signature_count = 0
        if not isinstance(arg, (list, tuple)):
            return False

        if hasattr(self, '_pd_number_of_inputs') and self._pd_number_of_inputs is not None:
            return self._pd_number_of_inputs > 1

        try:
            parameters = self.pd_get_signature().values()
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
        """Call transform with possibility to pass multiple arguments"""

        if stage in ('eval', 'infer'):
            torch_context = torch.no_grad()
        else:
            torch_context = contextlib.suppress()

        with self.pd_set_stage(stage), torch_context:
            if self._pd_unpack_argument(arg):
                return self(*arg)
            return self(arg)

    def pd_get_signature(self):
        return inspect.signature(self).parameters

    def _pd_callyield(self, args, stage: Stage, loader_kwargs: Optional[dict] = None,
                      verbose: bool = False, flatten: bool = False):  # TODO: different name?
        """Create a data loader and run preprocessing, forward, and postprocessing steps.

        :param args: Arguments to call with.
        :param stage: Stage to call in ("eval", "train" or "infer")
        :param loader_kwargs: Data loader keyword arguments.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.

        :returns: A generator that allows iterating over the output.
        """
        assert stage in ('eval', 'train'), '_pd_callyield can only be used with stage eval or train'

        self._pd_forward_device_check()

        preprocess = self.pd_preprocess
        forward = self.pd_forward
        post = self.pd_postprocess

        use_preprocess = not preprocess.pd_is_identity
        use_forward = not forward.pd_is_identity
        use_post = not post.pd_is_identity

        if use_preprocess:
            iterator = SimpleIterator(
                args,
                lambda *args: self.pd_preprocess._pd_call_transform(*args, stage)
            )
            if loader_kwargs is None:
                loader_kwargs = {}
            loader = self._pd_get_loader(iterator=iterator, loader_kwargs=loader_kwargs)
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
                output = forward._pd_call_transform(batch, stage)
            else:
                output = batch

            if use_post:
                output = post._pd_call_transform(output, stage)

            if flatten:
                if verbose:
                    pbar.update()
                if not use_post:
                    output = Unbatchify()(batch)
                yield from output
                continue

            yield output

    @property
    def pd_device(self) -> str:  # TODO: remove?
        """Return the device the transform is on."""
        return self._pd_device

    @property
    def pd_component(self) -> Set[Component]:
        """Return the component (preprocess, forward or postprocess)."""
        return self._pd_component

    @property
    def pd_preprocess(self) -> "Transform":
        """The preprocessing part. """
        if 'preprocess' in self.pd_component:
            return self
        return Identity()

    def _pd_forward_part(self) -> "Transform":
        """The forward part of the transform """
        if 'forward' in self.pd_component:
            return self
        return Identity()

    @property
    def pd_forward(self) -> "Transform":
        """The forward part of the transform and send to GPU"""
        return self._pd_forward_part()

    @property
    def pd_postprocess(self) -> "Transform":
        """The postprocessing part of the transform. """
        if 'postprocess' in self.pd_component:
            return self
        return Identity()

    def pd_to(self, device: str):  # TODO: change
        """Set the transform's device to *device*.

        :param device: device on which to map {'cpu', 'cuda', 'cuda:N'}
        """
        self._pd_device = device
        for layer in self.pd_layers:
            layer.to(device)
        return self

    @property
    def pd_is_identity(self):  # TODO: keep?
        """Return *True* iff the transform is the identity transform. """
        return False

    @property
    def pd_layers(self) -> List[torch.nn.Module]:
        """Get a dict with all layers in the transform (including layers in sub-transforms)."""
        layers = []
        for subtrans in self.pd_all_transforms():
            if isinstance(subtrans, torch.nn.Module):
                layers.append(subtrans)
        return layers

    def pd_parameters(self) -> Iterator:
        """ Iterate over parameters. """
        for layer in self.pd_layers:
            yield from layer.parameters()

    @contextlib.contextmanager
    def pd_set_stage(self, stage: Optional[str]=None):
        """Set of stage of Transform

        :param stage: stage ('train', 'eval', 'infer')
        """
        assert stage in ('train', 'eval', 'infer', None)

        if stage is None:
            yield
            return

        layers = self.pd_layers  # TODO: set back?
        try:
            for layer in layers:
                if stage == 'train':
                    layer.train()
                else:
                    layer.eval()
            Transform.pd_stage = stage
            yield
        # TODO: Should we put layers in eval mode by default?
        finally:
            for layer in layers:
                layer.eval()
            Transform.pd_stage = None

    @staticmethod
    def _pd_get_loader(iterator, loader_kwargs=None):
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

    def infer_apply(self, input):
        """Call transform within the infer context.

        This expects a single argument and returns a single output.
        """
        return self._pd_call_transform(input, stage='infer')

    def eval_apply(self, inputs: Iterable,
                   verbose: bool = False, flatten: bool = False, **kwargs):
        """Call transform within the eval context.

        This expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_callyield(inputs, 'eval', loader_kwargs=kwargs,
                                  verbose=verbose, flatten=flatten)

    def train_apply(self, inputs: Iterable,
                    verbose: bool = False, flatten: bool = False, **kwargs):
        """Call transform within the train context.

        This expects an iterable input and returns a generator.

        :param inputs: The arguments - an iterable (e.g. list) of inputs.
        :param kwargs: Keyword arguments to be passed on to the dataloader. These can be
            any that a `torch.data.utils.DataLoader` accepts.
        :param verbose: If *True*, print progress bar.
        :param flatten: If *True*, flatten the output.
        """
        return self._pd_callyield(inputs, 'train', loader_kwargs=kwargs,
                                  verbose=verbose, flatten=flatten)


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms - in contrast to `CompoundTransform`s).

    Examples of `AtomicTransform`s are `ClassTransform`s and `FunctionTransform`s.

    :param call:
    :param call_info:
    :param pd_name: name of the transform
    """

    def __init__(self, call: str, call_info: Optional[inspector.CallInfo] = None,
                 pd_name: Optional[str] = None):
        super().__init__(call_info, pd_name)
        self._pd_call = call

    def pd_evaluable_repr(self, indent=0, var_transforms=None):
        if var_transforms is None:
            var_transforms = {}
        return var_transforms.get(self, self._pd_call)

    def _pd_title(self):
        return self._pd_call

    @property
    def pd_direct_subtransforms(self):
        # pylint: disable=no-self-use
        globals_dict, nonlocals_dict = self._pd_closurevars
        for v in chain(self.__dict__.values(), globals_dict.values(), nonlocals_dict.values()):
            if isinstance(v, Transform):
                yield v


class FunctionTransform(AtomicTransform):
    """Function Transform

    :param function: function to call
    :param call_info:
    :param pd_name: name of the transform
    :param call:
    """
    def __init__(self, function, call_info, pd_name=None, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call=call, call_info=call_info, pd_name=pd_name)
        self.function = function
        self._pd_number_of_inputs = None

    @property
    def source(self, length=20):
        try:
            body_msg = inspect.getsource(self.function)
            body_msg = ''.join(re.split('(def )', body_msg, 1)[1:])
            lines = re.split('(\n)', body_msg)
            lines = ''.join(lines[:length]) + ('  ...' if len(lines) > length else '')
            if len(lines) == 0:
                body_msg = ''.join(re.split('(lambda)', self._pd_call, 1)[1:])[:-1]
                return body_msg[:100]
            return lines
        except TypeError:
            return self._pd_call

    def pd_get_signature(self):
        if self._pd_number_of_inputs is None:
            return inspect.signature(self).parameters
        return [f'arg_{i}' for i in range(self._pd_number_of_inputs)]

    def _pd_bodystr(self, length=20):
        return self.source

    def _pd_title(self):
        return self._pd_call.split('(')[0]

    @property
    def _pd_closurevars(self) -> inspect.ClosureVars:
        try:
            closurevars = inspect.getclosurevars(self.function)
        except TypeError as exc:
            warn(f"Couln't get closurevars ({exc}). This is usually fine.")
            return {}, {}
        return closurevars.globals, closurevars.nonlocals

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ClassTransform(AtomicTransform):
    """Class Transform.

    :param pd_name: Name of the transform.
    """

    def __init__(self, pd_name=None, ignore_scope=False, arguments=None):
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
    def source(self, length=40):
        try:
            (body_msg, _), _ = thingfinder.find_in_scope(self.__class__.__name__,
                                                    self._pd_call_info.scope)
            body_msg = ''.join(re.split('(class )', body_msg, 1)[1:])
            lines = re.split('(\n)', body_msg)
            return ''.join(lines[:length]) + ('  ...' if len(lines) > length else '')
        except thingfinder.ThingNotFound:
            return self._pd_call

    def _format_args(self):
        args_list = []
        for key, value in self._pd_arguments.items():
            if key == 'args':
                args_list += [f'{format_argument(val)}' for val in value]
            elif key == 'kwargs':
                args_list += [f'{subkey}={format_argument(val)}' for subkey, val in value.items()]
            else:
                args_list.append(f'{key}={format_argument(value)}')
        return ', '.join(args_list)

    def _pd_title(self):
        title = type(self).__name__
        return title + '(' + self._format_args() + ')'

    def _pd_bodystr(self):
        return self.source


class TorchModuleTransform(ClassTransform):
    """Torch Module Transform"""
    def pd_get_signature(self):
        return inspect.signature(self.forward).parameters

    def pd_pre_save(self, path, i):
        """
        :param path: The save path.
        :param i: Sublayer index.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def pd_post_load(self, path, i):
        """
        :param path: The load path.
        :param i: Sublayer index.
        """
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))

    def _pd_bodystr(self):
        return torch.nn.Module.__repr__(self)


class Map(Transform):
    """Apply one transform to each element of a list.

    >>> Map(t)([x1, x2, x3]) == [t(x1), t(x2), t(x3)]
    True

    :param transform: transform to be applied to a list of inputs
    """

    def __init__(self, transform, call_info=None, pd_name=None):
        super().__init__(call_info, pd_name)

        self.transform = transform
        self._pd_component = transform.pd_component

        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

    def __call__(self, arglist):
        """
        :param arglist: Args list to call transforms with
        """
        return [self.transform._pd_call_transform(arg) for arg in arglist]

    def _pd_bodystr(self):
        return '~ ' + self.transform._pd_shortname()

    def _pd_title(self):
        return self.__class__.__name__

    @property
    def pd_direct_subtransforms(self):
        yield self.transform

    def pd_evaluable_repr(self, indent=0, var_transforms=None):
        return f'~{self.transform.pd_evaluable_repr(indent + 4, var_transforms)}'

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
    def pd_preprocess(self):
        if self._pd_preprocess is None:
            t_pre = self.transform.pd_preprocess
            if isinstance(t_pre, Identity):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Map(transform=t_pre, call_info=self._pd_call_info)
        return self._pd_preprocess

    @property
    def pd_postprocess(self):
        if self._pd_postprocess is None:
            t_post = self.transform.pd_postprocess
            if isinstance(t_post, Identity):
                self._pd_postprocess = Identity()
            else:
                self._pd_postprocess = Map(transform=t_post, call_info=self._pd_call_info)
        return self._pd_postprocess

    def _pd_forward_part(self):
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
    :param call_info:
    :param pd_name: name of CompoundTransform
    :param pd_group:
    """
    op = NotImplemented

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(call_info, pd_name)

        self._pd_group = True if pd_name is not None else pd_group
        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

        transforms = self._flatten_list(transforms)
        self.transforms = transforms

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

    def pd_evaluable_repr(self, indent=0, var_transforms=None):
        sub_reprs = [
            x.pd_varname() or x.pd_evaluable_repr(indent + 4, var_transforms)
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
            graph['padl', scope] = var2mod.CodeNode.from_source('import padl', scope)
            scopemap['padl', scope] = scope

        if name is not None:
            assert scope is not None
            graph[name, scope] = start
            scopemap[name, scope] = scope

        for transform in self.transforms:
            varname = transform.pd_varname()
            transform._pd_build_codegraph(graph, scopemap, varname,
                                          self._pd_call_info.scope)
        return graph, scopemap

    def pd_repr(self, indent=0):
        sub_reprs = [
            x.pd_repr(indent + 4)
            for x in self.transforms
        ]
        res = (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )
        if self.pd_varname() is not None and self.pd_varname() is not _notset:
            res += f' [{self.pd_varname()}]'
        return res

    def _add_parentheses_if_needed(self, name):
        if self._pd_name is None:
            return '(' + name + ')'
        return name

    def _pd_shortname(self, bold_title=False):
        if self._pd_name is None:
            return self._pd_bodystr(is_child=True)
        return super()._pd_shortname(bold_title)

    def _pd_title(self):
        return self.__class__.__name__

    def _pd_bodystr(self, is_child=False):
        sep = f' {self.op} ' if is_child else '\n'
        if is_child:
            return sep.join(t._add_parentheses_if_needed(t._pd_shortname())
                            for t in self.transforms)
        return sep.join(t._pd_shortname() for t in self.transforms)

    def _pd_add_format_to_str(self, name):
        """
        Create formatted output based on "name" input lines. For multi-line inputs
        the lines are infixed with *self.display_op*

        :param name: line or lines of input
        """
        res = f'\n        {make_green(self.display_op)}  \n'.join(
            [make_bold(f'{i}: ') + x for i, x in enumerate(name.split('\n'))]) + '\n'
        return res

    def pd_to(self, device: str):
        """Set the transform's device to *device*

        :param device: device on which to send {'cpu', cuda', 'cuda:N'}
        """
        self._pd_device = device
        for transform_ in self.transforms:
            transform_.pd_to(device)
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
    def pd_direct_subtransforms(self):
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
            if transform_.pd_name is None:
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
    :param call_info:
    :param pd_name: name of the Compose transform
    :param pd_group:
    :return: output from series of transforms
    """
    op = '>>'
    display_op = '↓'

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
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

    @property
    def n_display_inputs(self):
        return self.transforms[0].n_display_inputs

    @property
    def n_display_outputs(self):
        return self.transforms[-1].n_display_outputs

    def __repr__(self) -> str:
        top_message = make_bold(Transform._pd_shortname(self) + ':') + '\n\n'
        return top_message + self._pd_write_arrows_to_rows()

    def _pd_write_arrows_to_rows(self):
        """
        Create formatted output based on "name" input lines. For multi-line inputs
        the lines are infixed with *self.display_op*

        :param name: line or lines of input
        """
        # pad the components of rows which are shorter than other parts in same column
        rows = [
            [s._pd_shortname().strip() for s in t.transforms]
            if isinstance(t, CompoundTransform)
            else [t._pd_shortname()]
            for t in self.transforms
        ]

        children_widths = [[len(x) for x in row] for row in rows]
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
                rows[i] = f' {self.transforms[i].display_op} '.join(r)
            else:
                rows[i] = r[0]
        output = []
        # iterate through rows and create arrows depending on numbers of components
        for i, (r, t) in enumerate(zip(rows, self.transforms)):
            widths = [0]
            subarrows = []

            # if subsequent rows have the same number of "children" transforms
            if i > 0 and len(children_widths[i]) == len(children_widths[i - 1]):
                for j, w in enumerate(children_widths[i]):
                    subarrows.append(create_arrow(sum(widths) - j + j * 4, 0, 0, 0))
                    widths.append(int(max_widths[j]))

            # if previous row has multiple outputs and current row just one input
            elif i > 0 and len(children_widths[i]) == 1 \
                    and len(children_widths[i - 1]) > 1:
                for j, w in enumerate(children_widths[i - 1]):
                    subarrows.append(create_reverse_arrow(
                        j, sum(widths) - j + j * 3,
                        len(children_widths[i - 1]) - j + 1, j + 1
                    ))
                    widths.append(int(max_widths[j]))

            # if previous row has one output and current row has multiple inputs
            else:
                for j, w in enumerate(children_widths[i]):
                    subarrows.append(create_arrow(j, sum(widths) - j + j * 3,
                                                  len(children_widths[i]) - j, j + 1))
                    widths.append(int(max_widths[j]))

            # add signature names to the arrows
            tuple_to_str = lambda x: '(' + ', '.join([str(y) for y in x]) + ')'
            if (isinstance(t, Rollout) or isinstance(t, Parallel)) and t._pd_name is None:
                all_params = []
                for tt in t.transforms:
                    all_params.append(list(tt.pd_get_signature().keys()))
                to_combine = [
                    ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + tuple_to_str(params)
                    if len(params) > 1
                    else ' ' * (sum(widths[:k + 1]) + 3 * k + 2) + params[0]
                    for k, params in enumerate(all_params)
                ]
                to_format = combine_multi_line_strings(to_combine)
            else:
                padder = (len(subarrows) + 1) * ' '
                params = [x for x in t.pd_get_signature()]
                to_format = padder + tuple_to_str(params) if len(params) > 1 else padder + params[0]
            to_format_pad_length = max([len(x.split('\n')) for x in subarrows]) - 1
            to_format = ''.join(['\n' for _ in range(to_format_pad_length)] + [to_format])

            # combine the arrows
            mark = combine_multi_line_strings(subarrows + [to_format])
            mark = '\n'.join(['   ' + x for x in mark.split('\n')])
            output.append(make_green(mark))
            output.append(make_bold(f'{i}: ') + r)
        return '\n'.join(output)

    def __call__(self, arg):
        """Call method for Compose

        :param arg: arguments to call with
        :return: output from series of transforms
        """
        for transform_ in self.transforms:
            arg = transform_._pd_call_transform(arg)
        return arg

    def _pd_forward_part(self):
        """Forward part of transforms"""
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
    def pd_preprocess(self):
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
    def pd_postprocess(self):
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
    :param call_info:
    :param pd_name: name of the Rollout transform
    :param pd_group:
    :return: namedtuple of outputs
    """
    op = '+'
    display_op = '+'

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    @property
    def n_display_inputs(self):
        return len(self.transforms)

    @property
    def n_display_outputs(self):
        return len(self.transforms)

    @property
    def children_widths(self):
        return [t.display_width for t in self.transforms]

    def __call__(self, arg):
        """Call method for Rollout

        :param arg: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for transform_ in self.transforms:
            out.append(transform_._pd_call_transform(arg))
        out = self._pd_output_format(*out)
        return out

    @property
    def pd_preprocess(self):
        if self._pd_preprocess is None:
            t_list = [x.pd_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Rollout(t_list, call_info=self._pd_call_info)
        return self._pd_preprocess

    def _pd_forward_part(self):
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
    def pd_postprocess(self):
        if self._pd_postprocess is None:
            t_list = [x.pd_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_postprocess = Identity()
            elif len(list(self._pd_component)) >= 2 and 'postprocess' in self._pd_component:
                self._pd_postprocess = Parallel(t_list, call_info=self._pd_call_info)
            else:
                self._pd_postprocess = Rollout(t_list, call_info=self._pd_call_info)
        return self._pd_postprocess


class Parallel(CompoundTransform):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param call_info:
    :param pd_name: name of the Parallel transform
    :param pd_group:
    :return: namedtuple of outputs
    """
    op = '/'
    display_op = '/'

    def __init__(self, transforms, call_info=None, pd_name=None, pd_group=False):
        super().__init__(transforms, call_info=call_info, pd_name=pd_name, pd_group=pd_group)
        self.pd_keys = self._pd_get_keys(self.transforms)
        self._pd_output_format = namedtuple('namedtuple', self.pd_keys)

    @property
    def n_display_inputs(self):
        return len(self.transforms)

    @property
    def children_widths(self):
        return [t.display_width for t in self.transforms]

    def __call__(self, arg):
        """Call method for Parallel

        :param arg: Argument to call with.
        :return: Namedtuple of output.
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            out.append(transform_._pd_call_transform(arg[ind]))
        out = self._pd_output_format(*out)
        return out

    @property
    def pd_preprocess(self):
        if self._pd_preprocess is None:
            t_list = [x.pd_preprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_preprocess = Identity()
            else:
                self._pd_preprocess = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_preprocess

    def _pd_forward_part(self):
        if self._pd_forward is None:
            t_list = [x.pd_forward for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_forward = Identity()
            else:
                self._pd_forward = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_forward

    @property
    def pd_postprocess(self):
        if self._pd_postprocess is None:
            t_list = [x.pd_postprocess for x in self.transforms]
            if all([isinstance(t, Identity) for t in t_list]):
                self._pd_postprocess = Identity()
            else:
                self._pd_postprocess = Parallel(t_list, call_info=self._pd_call_info)
        return self._pd_postprocess


class BuiltinTransform(AtomicTransform):
    def __init__(self, call):
        super().__init__(call)

    def _pd_build_codegraph(self, graph: Optional[dict] = None,
                            scopemap: Optional[dict] = None,
                            name: Optional[str] = None,
                            scope: Optional[thingfinder.Scope] = None) -> Tuple[dict, dict]:
        if graph is None:
            graph = {}
        if scopemap is None:
            scopemap = {}

        if scope is None:
            scope = thingfinder.Scope.empty()
        if ('padl', scope) not in graph:
            graph['padl', scope] = var2mod.CodeNode.from_source('import padl', scope)
            scopemap['padl', scope] = scope

        start_source = f'{name or "_pd_dummy"} = {self._pd_call}'

        graph[self.__class__.__name__, scope] = \
            var2mod.CodeNode.from_source(start_source, scope)

        scopemap[self.__class__.__name__, scope] = scope

        return graph, scopemap

    def _pd_bodystr(self):
        return self._pd_call.split('padl.')[-1]

    def _pd_title(self):
        return self._pd_call.split('padl.')[-1].split('(')[0]


class Identity(BuiltinTransform):
    """Do nothing.

    :param pd_name: name of the transform
    """

    def __init__(self):
        super().__init__('padl.Identity()')

    @property
    def pd_is_identity(self):
        return True

    def __call__(self, args):
        return args


class Unbatchify(ClassTransform):
    """Mark start of postprocessing

    Unbatchify removes batch dimension (inverse of Batchify) and moves the input tensors to 'cpu'.

    :param dim: batching dimension
    :param cpu: if true, moves output to cpu after unbatchify
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
        assert Transform.pd_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

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

    :param dim: batching dimension
    """

    def __init__(self, dim=0):
        super().__init__(arguments=OrderedDict([('dim', dim)]))
        self.dim = dim
        self._pd_component = {'preprocess'}

    def _move_to_device(self, args):
        if isinstance(args, (tuple, list)):
            return tuple([self._move_to_device(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.to(self.pd_device)
        return args

    def __call__(self, args):
        assert Transform.pd_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if Transform.pd_stage != 'infer':
            return self._move_to_device(args)
        if isinstance(args, (tuple, list)):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim).to(self.pd_device)
        if isinstance(args, (float, int)):
            return torch.tensor([args]).to(self.pd_device)
        raise TypeError('only tensors and tuples of tensors recursively supported...')


def save(transform: Transform, path):
    transform.pd_save(path)


def load(path):
    """Load transform (as saved with padl.save) from *path*. """
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
    for i, subtrans in enumerate(transform.pd_all_transforms()):
        subtrans.pd_post_load(path, i)
    return transform


def group(transform: Union[Rollout, Parallel]):
    return transform.grouped()
