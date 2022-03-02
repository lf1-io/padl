"""Wrappers. """

import ast
import dis
import functools
import inspect
import copy
from types import MethodWrapperType, ModuleType
import importlib

import numpy as np
import torch

from padl.dumptools import ast_utils, var2mod, inspector
from padl.dumptools.sourceget import cut, get_source, original
from padl.transforms import (
    AtomicTransform, ClassTransform, FunctionTransform, TorchModuleTransform
)

import re

def _set_local_varname(frame, event, _args, scope):
    if event == 'return':
        for k, v in frame.f_locals.items():
            try:
                v._pd_varname[scope] = k
            except AttributeError:
                continue


def _wrap_function(fun, ignore_scope=False, call_info: inspector.CallInfo = None):
    """Wrap *fun* in a Transform. Don't use directly, use `transform` instead.

    :param fun: function to be wrapped
    :param call_info: A `CallInfo` object containing information about the how the transform was
        created (needed for saving).
    """
    caller = inspect.stack()[2]

    if '@' in caller.code_context[0] or caller.code_context[0].startswith('def'):
        call = None
        wrap_type = 'decorator'
    else:
        try:
            # case transform(f)
            call = inspector.get_segment_from_frame(caller.frame, 'call')
            wrap_type = 'inline'
        except (RuntimeError, IndexError):
            # case transform(some_module).f
            try:
                call = inspector.get_segment_from_frame(caller.frame, 'attribute')
                wrap_type = 'module'
            except RuntimeError:
                # needed for python 3.7 support
                call = None
                wrap_type = 'decorator'

    # if this is the decorator case we drop one leven from the scope (this is the decorated
    # function itself)
    drop_n = 1 if call is None else 0
    if call_info is None:
        call_info = inspector.CallInfo(drop_n=drop_n, ignore_scope=ignore_scope)
    if call_info.function != '<module>' and not ignore_scope:
        inspector.trace_this(_set_local_varname, caller.frame, scope=call_info.scope)

    wrapper = FunctionTransform(fun, call_info, call=call, wrap_type=wrap_type)

    # Special checks
    if isinstance(fun, np.ufunc):
        wrapper._pd_number_of_inputs = fun.nin

    functools.update_wrapper(wrapper, fun)
    return wrapper


def _wrap_class(cls, ignore_scope=False):
    """Patch __init__ of class such that the initialization statement is stored
    as an attribute `_pd_call`. In addition make class inherit from Transform.

    This is called by `transform`, don't call `_wrap_class` directly, always use `transform`.

    Example:
    >>> @transform
    ... class MyClass:
    ...     def __init__(self, x):
    ...         self.x = x
    ...     def __call__(self, args):
    ...         return self.x + args
    >>> myobj = MyClass('hello')
    >>> myobj._pd_call
    "MyClass('hello')"

    :param cls: class to be wrapped
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    """
    old__init__ = cls.__init__
    if issubclass(cls, torch.nn.Module):
        trans_class = TorchModuleTransform
    else:
        trans_class = ClassTransform

    module = cls.__module__
    # make cls inherit from AtomicTransform
    cls = type(cls.__name__, (trans_class, cls), {})

    signature = inspect.signature(old__init__)

    @functools.wraps(cls.__init__)
    def __init__(self, *args, **kwargs):
        old__init__(self, *args, **kwargs)
        args = signature.bind(None, *args, **kwargs).arguments
        args.pop(next(iter(args.keys())))
        trans_class.__init__(self, ignore_scope=ignore_scope, arguments=args)

    functools.update_wrapper(__init__, old__init__)

    cls.__init__ = __init__
    cls.__module__ = module
    cls._pd_class_call_info = inspector.CallInfo()
    return cls


def _wrap_class_instance(obj, ignore_scope=False):
    """Patch __class__ of a class instance such that inherits from Transform.

    This is called by `transform`, don't call `_wrap_class_instance` directly, always use
    `transform`.

    Example:
    >>> class MyClass:
    ...     def __init__(self, x):
    ...         self.x = x
    ...     def __call__(self, args):
    ...         return self.x + args
    >>> myobj = transform(MyClass('hello'))

    :param obj: object to be wrapped
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    """
    if issubclass(type(obj), torch.nn.Module):
        trans_class = TorchModuleTransform
    else:
        trans_class = ClassTransform

    obj_copy = copy.copy(obj)
    obj_copy.__class__ = type(type(obj).__name__, (trans_class, type(obj)), {})

    caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
    call_info = inspector.CallInfo(caller_frameinfo, ignore_scope=ignore_scope)
    call = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call')
    call = re.sub(r'\n\s*', ' ', call)
    obj_copy._pd_arguments = None

    AtomicTransform.__init__(obj_copy, call=call, call_info=call_info)

    return obj_copy


def _wrap_lambda(fun, ignore_scope=False):
    """Wrap a lambda function in a transform. Hacky hack that will hopefully
    become obsolete with python 3.11 (see also inspector.CallInfo).

    :param fun: function to be wrapped
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    """
    # get the caller frame (it's 2 - [caller] -> [trans] -> [_wrap_lambda])
    caller_frame = inspector.caller_frame()
    # get the source
    try:
        full_source = caller_frame.f_globals['_pd_source']
    except KeyError:
        full_source = get_source(caller_frame.f_code.co_filename)

    source, offset = inspector.get_statement(original(full_source), caller_frame.f_lineno)
    # find all lambda nodes
    nodes = var2mod.Finder(ast.Lambda).find(ast.parse(source))
    candidate_segments = []
    candidate_calls = []
    for node in nodes:
        # keep lambda nodes which are contained in a call of `lf.trans`
        if not isinstance(node.parent, ast.Call):
            continue
        containing_call = ast_utils.get_source_segment(source, node.parent.func)
        containing_function = eval(containing_call, caller_frame.f_globals)
        if containing_function is not transform:
            continue
        candidate_segments.append(
            ast_utils.get_source_segment(source, node),
        )
        candidate_calls.append((
            ast_utils.get_source_segment(source, node.parent),
            ast_utils.get_position(source, node.parent)
        ))

    # compare candidate's bytecodes to that of `fun`
    # keep the call for the matching one
    target_instrs = list(dis.get_instructions(fun))

    found = False
    call = None
    locs = None

    for segment, (call, locs) in zip(candidate_segments, candidate_calls):
        instrs = list(dis.get_instructions(eval(segment)))
        if not len(instrs) == len(target_instrs):
            continue
        for instr, target_instr in zip(instrs, target_instrs):
            if instr.argval != target_instr.argval:
                break
            same_opname = instr.opname == target_instr.opname
            load_ops = ('LOAD_NAME', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_CONST', 'LOAD_DEREF')
            both_load = instr.opname in load_ops and target_instr.opname in load_ops
            if not (same_opname or both_load):
                break
        else:
            found = True
            break

    if call is None or not found:
        raise RuntimeError('Lambda not found.')

    locs = (
        locs.lineno - 1 + offset[0],
        locs.end_lineno - 1 + offset[0],
        locs.col_offset - offset[1],
        locs.end_col_offset - offset[1]
    )

    call = cut(full_source, *locs)

    caller = inspector.CallInfo(ignore_scope=ignore_scope)
    inner = var2mod.Finder(ast.Lambda).get_source_segments(call)[0][0]
    wrapper = FunctionTransform(fun, caller, call=call, source=inner, wrap_type='lambda')
    functools.update_wrapper(wrapper, fun)
    return wrapper


class PatchedModule:
    """Class that patches a module, such that all functions and classes in that module come out
    wrapped as Transforms.

    Example:

    >>> import padl
    >>> import numpy as np
    >>> pd_np = padl.transform(np)
    >>> isinstance(pd_np.random.rand, padl.transforms.Transform)
    True
    """

    def __init__(self, module, parents=None):
        self._module = module
        if parents is None:
            self._path = self._module.__name__
        else:
            self._path = parents + '.' + self._module.__name__

    def __getattr__(self, key):
        x = getattr(self._module, key)
        if inspect.isclass(x):
            if hasattr(x, '__call__') and not isinstance(x.__call__, MethodWrapperType):
                return _wrap_class(x)
            return x
        if callable(x):
            call_info = inspector.CallInfo()
            return _wrap_function(x, ignore_scope=True, call_info=call_info)
        if isinstance(x, ModuleType):
            return PatchedModule(x, parents=self._path)
        return x

    def __repr__(self):
        return f'Transform patched: {self._module}'

    def __dir__(self):
        return dir(self._module)


def _wrap_module(module):
    module = importlib.import_module(module.__name__)
    return PatchedModule(module)


def transform(wrappee, ignore_scope=False):
    """Transform wrapper / decorator. Use to wrap a class, module or callable.

    :param wrappee: class, module or callable to be wrapped
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    """
    if isinstance(wrappee, ModuleType):
        return _wrap_module(wrappee)
    if inspect.isclass(wrappee):
        return _wrap_class(wrappee, ignore_scope)
    if callable(wrappee):
        if not hasattr(wrappee, '__name__'):
            return _wrap_class_instance(wrappee, ignore_scope)
        if wrappee.__name__ == '<lambda>':
            return _wrap_lambda(wrappee, ignore_scope)
        return _wrap_function(wrappee, ignore_scope)
    raise ValueError('Can only wrap classes or callables.')
