"""Wrappers. """
import ast
import dis
import functools
import inspect

import torch

from lf.dumptools import var2mod, inspector
from lf.transform import AtomicTransform, FunctionTransform, TorchModuleTransform, _notset


def _set_local_varname(frame, event, _args):
    if event == 'return':
        for k, v in frame.f_locals.items():
            try:
                if v._lf_varname is _notset or v._lf_varname is None:
                    v.set_lf_varname(k)
            except AttributeError:
                continue


def _wrap_function(fun):
    """Fram *fun* in a Transform. Don't use directly, use `trans` instead. """
    call_info = inspector.caller_info(1)
    caller = inspect.stack()[2]
    if call_info.function != '<module>':
        inspector.trace_this(_set_local_varname, caller.frame)
    closurevars = inspect.getclosurevars(fun)
    call_info.globals = closurevars.globals
    call_info.nonlocals = closurevars.nonlocals
    wrapper = FunctionTransform(fun, call_info)
    functools.update_wrapper(wrapper, fun)
    return wrapper


def _wrap_class(cls):
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
        trans_class = AtomicTransform

    # make cls inherit from AtomicTransform
    cls = type(cls.__name__, (cls, trans_class), {})

    @functools.wraps(cls.__init__)
    def __init__(self, *args, **kwargs):
        old__init__(self, *args, **kwargs)
        trans_class.__init__(
            self,
            inspector.get_call_segment_from_frame(inspector.caller()),
            inspector.caller_info()
        )

    cls.__init__ = __init__
    return cls


def _wrap_lambda(fun):
    """Wrap a lambda function in a transform. Hacky hack that will hopefully
    become obsolete with python 3.11 (see _wrap_class). """
    # get the caller frame (it's 2 - [caller] -> [trans] -> [_wrap_lambda])
    caller_frame = inspector.caller()
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

    wrapper = FunctionTransform(fun, inspector.caller_info(), call)
    functools.update_wrapper(wrapper, fun)
    return wrapper


def trans(fun_or_cls):
    """Transform wrapper / decorator. Use to wrap a class or callable. """
    if inspect.isclass(fun_or_cls):
        return _wrap_class(fun_or_cls)
    if inspect.isfunction(fun_or_cls):
        if fun_or_cls.__name__ == '<lambda>':
            return _wrap_lambda(fun_or_cls)
        return _wrap_function(fun_or_cls)
    raise ValueError('Can only wrap classes or callables.')
