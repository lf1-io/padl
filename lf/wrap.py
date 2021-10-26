"""Wrappers. """
import ast
import dis
import functools
import inspect
from warnings import warn

import torch

from lf.dumptools import var2mod, inspector
from lf.transform import (
    AtomicTransform, ClassTransform, FunctionTransform, TorchModuleTransform, _notset
)


def _set_local_varname(frame, event, _args):
    if event == 'return':
        for k, v in frame.f_locals.items():
            try:
                if v._lf_varname is _notset or v._lf_varname is None:
                    v._lf_set_varname(k)
            except AttributeError:
                continue


def _wrap_function(fun, ignore_scope=False):
    """Fram *fun* in a Transform. Don't use directly, use `trans` instead. """
    call_info = inspector.CallInfo(drop_n=1, ignore_scope=ignore_scope)
    caller = inspect.stack()[2]
    if call_info.function != '<module>' and not ignore_scope:
        inspector.trace_this(_set_local_varname, caller.frame)

    try:
        call = inspector.get_attribute_segment_from_frame(caller.frame)
    except RuntimeError:
        call = None
    wrapper = FunctionTransform(fun, call_info, call=call)
    functools.update_wrapper(wrapper, fun)
    return wrapper


def _wrap_class(cls, ignore_scope=False):
    """Patch __init__ of class such that the initialization statement is stored
    as an attribute `_lf_call`. In addition make class inherit from Transform.

    This is called by `trans`, don't call `_wrap_class` directly, always use `trans`.

    Example:

    @trans
    class MyClass:
        def __init__(self, x):
            ...

    >>> myobj = MyClass('hello')
    >>> myobj._lf_call
    MyClass('hello')
    """
    old__init__ = cls.__init__
    if issubclass(cls, torch.nn.Module):
        trans_class = TorchModuleTransform
    else:
        trans_class = ClassTransform

    module = cls.__module__
    # make cls inherit from AtomicTransform
    cls = type(cls.__name__, (cls, trans_class), {})

    @functools.wraps(cls.__init__)
    def __init__(self, *args, **kwargs):
        old__init__(self, *args, **kwargs)
        trans_class.__init__(self, ignore_scope=ignore_scope)

    cls.__init__ = __init__
    cls.__module__ = module
    return cls


def _wrap_lambda(fun, ignore_scope=False):
    """Wrap a lambda function in a transform. Hacky hack that will hopefully
    become obsolete with python 3.11 (see _wrap_class). """
    # get the caller frame (it's 2 - [caller] -> [trans] -> [_wrap_lambda])
    caller_frame = inspector.caller_frame()
    # get the source
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = inspector.get_source(caller_frame.f_code.co_filename)
    source = inspector.get_statement(full_source, caller_frame.f_lineno)
    # find all lambda nodes
    nodes = var2mod.Finder(ast.Lambda).find(ast.parse(source))
    candidate_segments = []
    candidate_calls = []
    for node in nodes:
        # keep lambda nodes which are contained in a call of `lf.trans`
        if not isinstance(node.parent, ast.Call):
            continue
        containing_call = ast.get_source_segment(source, node.parent.func)
        containing_function = eval(containing_call, caller_frame.f_globals)
        if containing_function is not trans:
            continue
        candidate_segments.append(ast.get_source_segment(source, node))
        candidate_calls.append(ast.get_source_segment(source, node.parent))

    # compare candidate's bytecodes to that of `fun`
    # keep the call for the matching one
    target_instrs = list(dis.get_instructions(fun))

    call = None
    for segment, call in zip(candidate_segments, candidate_calls):
        instrs = list(dis.get_instructions(eval(segment)))
        if not len(instrs) == len(target_instrs):
            continue
        for instr, target_instr in zip(instrs, target_instrs):
            if (instr.opname, target_instr.argval) != (instr.opname, target_instr.argval):
                break
        else:
            break

    if call is None:
        raise RuntimeError('Lambda not found.')

    wrapper = FunctionTransform(fun, inspector.CallInfo(ignore_scope=ignore_scope), call=call)
    functools.update_wrapper(wrapper, fun)
    return wrapper


def trans(fun_or_cls, ignore_scope=False):
    """Transform wrapper / decorator. Use to wrap a class or callable. """
    if inspect.isclass(fun_or_cls):
        return _wrap_class(fun_or_cls, ignore_scope)
    if callable(fun_or_cls):
        if fun_or_cls.__name__ == '<lambda>':
            return _wrap_lambda(fun_or_cls, ignore_scope)
        return _wrap_function(fun_or_cls, ignore_scope)
    raise ValueError('Can only wrap classes or callables.')
