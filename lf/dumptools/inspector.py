"""Inspect utils. """
import ast
from dataclasses import dataclass, field
import dis
import inspect
import linecache
import sys
import types
from typing import Callable, Optional
from warnings import warn

from lf.dumptools import thingfinder, var2mod


@dataclass
class _CallInfo:
    # the module from which the call was made
    module: types.ModuleType
    function: str = '<module>'
    scope: Optional[thingfinder.Scope] = None
    # function definition source
    fdef_source: Optional[str] = None
    globals: dict = field(default_factory=dict)
    nonlocals: dict = field(default_factory=dict)


def caller_info(caller=None, drop_n=0):
    """Collect some information about the caller.

    :param drop_n: Drop *n* from the calling scope.

    :returns: A _CallInfo object.
    """
    if caller is None:
        calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
        caller = outer_caller(calling_module_name)
    module = _module(caller.frame)
    call_info = _CallInfo(module)
    call_info.function = caller.function
    if caller.function != '<module>':
        try:
            call_source = get_call_segment_from_frame(caller.frame.f_back)
            call_info.fdef_source = get_source(caller.filename)
            fdef_lineno = caller.frame.f_lineno
            call_info.scope = thingfinder.Scope.from_source(call_info.fdef_source, fdef_lineno,
                                                            call_source, call_info.module, drop_n)
            assert len(call_info.scope) <= 1, 'scope longer than 1 currently not supported'
        except (SyntaxError, IndexError) as exc:
            warn(f'Error determining scope, using top level: {exc}')  # TODO: fix this
            call_info.scope = thingfinder.Scope(module, '', [])
    else:
        call_info.scope = thingfinder.Scope(module, '', [])
    return call_info


def non_init_caller():
    stack = inspect.stack()
    for frameinfo in stack[1:]:
        if frameinfo.function != '__init__':
            break
    assert frameinfo.function != '__init__'
    return frameinfo


def trace_this(tracefunc: Callable, frame: Optional[types.FrameType] = None):
    """Call in a function body to trace the rest of the function execution with function
    *tracefunc*. *tracefunc* must match the requirements for the argument of `sys.settrace`
    (in the documentation of which more details can be found).

    Example:

    ```
    def tracefunc(frame, event, arg)
        if 'event' == 'return':
            print('returning', arg)

    def myfunction():
        [...]
        _trace_this(tracefunc)
        return 123
    ```

    :param tracefunc: Trace function (see documentation of `sys.settrace` for details).
    :param frame: The frame to trace (defaults to the caller's frame).
    """
    previous_tracefunc = sys.gettrace()

    if frame is None:
        # default is the caller's frame
        frame = inspect.currentframe().f_back

    def trace(frame, event, arg):
        tracefunc(frame, event, arg)
        if event == 'return':
            sys.settrace(previous_tracefunc)
        if previous_tracefunc is not None:
            previous_tracefunc(frame, event, arg)

    if previous_tracefunc is None:
        # set global tracefunc to something, this is required to enable local tracing
        sys.settrace(lambda _a, _b, _c: None)

    frame.f_trace = trace


def _instructions_up_to_call(x):
    """Get all instructions up to last CALL FUNCTION. """
    instructions = []
    instructions = list(dis.get_instructions(x))
    for i, instruction in enumerate(instructions[::-1]):
        if instruction.opname.startswith('CALL_'):
            break
    return instructions[:-i]


def _instructions_up_to_offset(x, lasti):
    """Get all instructions up to offset *lasti*. """
    instructions = []
    for instruction in dis.get_instructions(x):
        instructions.append(instruction)
        if instruction.offset == lasti:
            break
    return instructions


def get_source(filename):
    """Get source from *filename*.

    Filename as in the code object, can be "<ipython input-...>" in which case
    the source is taken from the ipython cache.
    """
    try:
        # the ipython case
        return ''.join(linecache.cache[filename][2])
    except KeyError:
        # normal module
        with open(filename) as f:
            return f.read()


def get_statement(source, lineno):
    """Get complete (potentially multi-line) statement at line *lineno* out of *source*. """
    module = ast.parse(source)
    stmts = []
    for stmt in module.body:
        if stmt.lineno <= lineno <= stmt.end_lineno:
            stmts.append(ast.get_source_segment(source, stmt))
    return '\n'.join(stmts)


def _module(frame):
    """Get module of *frame*. """
    try:
        return frame.f_globals['_lf_module']
    except KeyError:
        return sys.modules[frame.f_globals['__name__']]


def _same_module_stack(depth):
    stack = inspect.stack()[depth + 1:]
    module_name = stack[0].frame.f_globals['__name__']
    same_module = []
    for f in stack:
        if f.frame.f_globals['__name__'] == module_name:
            same_module.append(f)
        else:
            break
    res = []
    path = module_name
    for frame_info in same_module[::-1]:
        if frame_info.function != '<module>':
            path += '.' + frame_info.function
        res.append((frame_info.frame, path + ''))
    return res[::-1]


def outer_caller(module_name: str):
    """Get the first level of the stack before entering the module with name *module_name*. """
    stack = inspect.stack()
    before = True
    for frameinfo in stack:
        if frameinfo.frame.f_globals['__name__'] == module_name:
            before = False
            continue
        if before:
            continue
        return frameinfo


def caller_module():
    """Get the first module of the caller. """
    calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
    return _module(outer_caller(calling_module_name).frame)


def caller_frame():
    """Get the callers frame. """
    calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
    return outer_caller(calling_module_name).frame


def get_call_segment_from_frame(caller_frame):
    # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
    # , for python 3.11 (currently in development) this can be done via co_positions
    # (see https://www.python.org/dev/peps/pep-0657/),
    # for now, as 3.11 isn't widely used, this requires the following hack:
    # extract the source of the class init statement
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = get_source(caller_frame.f_code.co_filename)
    source = get_statement(full_source,
                           caller_frame.f_lineno)
    # the source can contain surrounding stuff we need to discard
    # as we only have the line number (this is what makes this complicated)

    # get all segments in the source that correspond to calls and might thus
    # potentially be the class init
    candidate_segments = var2mod.Finder(ast.Call).get_source_segments(source)
    # disassemble and get the instructions up to the current position
    target_instrs = _instructions_up_to_offset(caller_frame.f_code,
                                               caller_frame.f_lasti)
    # for each candidate, disassemble and compare the instructions to what we
    # actually have, a match means this is the correct statement
    if not candidate_segments:
        raise RuntimeError('No calls found.')

    segment = None

    for segment in candidate_segments:

        instrs = _instructions_up_to_call(segment)
        if len(instrs) > len(target_instrs):
            continue
        for instr, target_instr in zip(instrs, target_instrs[-len(instrs):]):
            if instr.argval != target_instr.argval:
                break
            same_opname = instr.opname == target_instr.opname
            load_ops = ('LOAD_NAME', 'LOAD_FAST', 'LOAD_GLOBAL')
            both_load = instr.opname in load_ops and target_instr.opname in load_ops
            if not (same_opname or both_load):
                break
        else:
            break

    if segment is None:
        raise RuntimeError('Call not found.')

    return segment
