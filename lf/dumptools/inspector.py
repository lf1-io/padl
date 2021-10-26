"""Inspect utils. """
import ast
import dis
import inspect
import linecache
import sys
import types
from typing import Callable, Optional
from warnings import warn

from lf.dumptools import thingfinder, var2mod


class CallInfo:
    """Information about the calling context.

    Contains the following information:
        - the module from which the call was made
        - the function from which that call was made
        - the scope (see `thingfinder.Scope`)

    :param drop_n: Drop *n* from the calling scope.

    :returns: A CallInfo object.
    """

    def __init__(self, origin='nextmodule', drop_n: int = 0, ignore_scope=False):
        assert isinstance(origin, inspect.FrameInfo) or origin in ('nextmodule', 'here')
        if isinstance(origin, inspect.FrameInfo):
            caller_frameinfo = origin
        if origin == 'nextmodule':
            calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
            caller_frameinfo = outer_caller_frameinfo(calling_module_name)
        elif origin == 'here':
            caller_frameinfo = inspect.stack()[1]
        self.function = caller_frameinfo.function
        self.scope = self._determine_scope(caller_frameinfo, drop_n, ignore_scope)

    def _determine_scope(self, caller_frameinfo: inspect.FrameInfo,
                         drop_n: int, ignore_scope: bool) -> thingfinder.Scope:
        module = _module(caller_frameinfo.frame)

        if self.function == '<module>' or ignore_scope:
            return thingfinder.Scope.toplevel(module)
        try:
            call_source = get_call_segment_from_frame(caller_frameinfo.frame.f_back)
            definition_source = get_source(caller_frameinfo.filename)
            fdef_lineno = caller_frameinfo.frame.f_lineno
            scope = thingfinder.Scope.from_source(definition_source, fdef_lineno,
                                                  call_source, module, drop_n)
            assert len(scope) <= 1, 'scope longer than 1 currently not supported'
            return scope
        except (SyntaxError, IndexError, RuntimeError) as exc:
            warn(f'Error determining scope, using top level: {exc}')  # TODO: fix this
            return thingfinder.Scope.toplevel(module)

    @property
    def module(self):
        """The calling module. """
        return self.scope.module


def non_init_caller_frameinfo() -> inspect.FrameInfo:  # TODO: generalize?
    """Get the FrameInfo for the first outer frame that is not of an "__init__" method. """
    stack = inspect.stack()
    frameinfo = None
    for frameinfo in stack[1:]:
        if frameinfo.function != '__init__':
            break
    assert frameinfo is not None and frameinfo.function != '__init__'
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
    instructions = list(dis.get_instructions(x))
    for i, instruction in enumerate(instructions[::-1]):
        if instruction.opname.startswith('CALL_'):
            break
    return instructions[:-i]


def _instructions_in_name(x):
    """Get all instructions up to last CALL FUNCTION. """
    instructions = list(dis.get_instructions(x))
    for i, instruction in enumerate(instructions):
        if instruction.opname not in ('LOAD_NAME', 'LOAD_ATTR'):
            break
    return instructions[:i]


def _instructions_up_to_offset(x, lasti):
    """Get all instructions up to offset *lasti*. """
    instructions = []
    for instruction in dis.get_instructions(x):
        instructions.append(instruction)
        if instruction.offset == lasti:
            break
    return instructions


def get_source(filename: str):
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


def get_statement(source: str, lineno: int):
    """Get complete (potentially multi-line) statement at line *lineno* out of *source*. """
    for i in range(lineno):
        try:
            block, lineno_in_block = get_surrounding_block(source, lineno - i)
        except ValueError:
            continue
        try:
            return _get_statement_from_block(block, lineno_in_block + i)
        except SyntaxError:
            continue
    raise SyntaxError("Couldn't find the statement.")


def _get_statement_from_block(block: str, lineno_in_block: int):
    module = ast.parse(block)
    stmts = []
    for stmt in module.body:
        if stmt.lineno <= lineno_in_block <= stmt.end_lineno:
            stmts.append(ast.get_source_segment(block, stmt))
    return '\n'.join(stmts)


def get_surrounding_block(source: str, lineno: int):
    """Get the code block surrounding the line at *lineno* in *source*.

    The code block surrounding a line is the largest block of lines with the same or larger
    indentation as the line itself.

    Raises a `ValueError` if the line at *lineno* is empty.

    :param source: The source to extract the block from.
    :param lineno: Number of the line for extracting the block.
    :returns: A tuple containing the block itself and the line number of the target line
        within the block.
    """
    lines = source.split('\n')
    before, after = lines[:lineno-1], lines[lineno:]
    white = thingfinder._count_leading_whitespace(lines[lineno-1])
    if white is None:
        raise ValueError('Line is empty.')
    block = [lines[lineno-1][white:]]
    lineno_in_block = 1
    while before:
        next_ = before.pop(-1)
        next_white = thingfinder._count_leading_whitespace(next_)
        if next_white is None or next_white >= white:
            block = [next_[white:]] + block
        else:
            break
        lineno_in_block += 1
    while after:
        next_ = after.pop(0)
        next_white = thingfinder._count_leading_whitespace(next_)
        if next_white is None or next_white >= white:
            block = block + [next_[white:]]
        else:
            break
    return '\n'.join(block), lineno_in_block


def _module(frame: types.FrameType):
    """Get module of *frame*. """
    try:
        return frame.f_globals['_lf_module']
    except KeyError:
        return sys.modules[frame.f_globals['__name__']]


def _same_module_stack(depth: int):  # TODO: remove (not being used)?
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


def outer_caller_frameinfo(module_name: str) -> inspect.FrameInfo:
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


def caller_module() -> types.ModuleType:
    """Get the first module of the caller. """
    calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
    return _module(outer_caller_frameinfo(calling_module_name).frame)


def caller_frame() -> types.FrameType:
    """Get the callers frame. """
    calling_module_name = inspect.currentframe().f_back.f_globals['__name__']
    return outer_caller_frameinfo(calling_module_name).frame


def get_call_segment_from_frame(caller_frame: types.FrameType) -> str:
    # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
    # , for python 3.11 (currently in development) this can be done via co_positions
    # (see https://www.python.org/dev/peps/pep-0657/),
    # for now, as 3.11 isn't widely used, this requires the following hack:
    # extract the source of the class init statement
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = get_source(caller_frame.f_code.co_filename)
    source = get_statement(full_source, caller_frame.f_lineno)
    # the source can contain surrounding stuff we need to discard
    # as we only have the line number (this is what makes this complicated)

    # get all segments in the source that correspond to calls and might thus
    # potentially be the class init
    candidate_segments = var2mod.Finder(ast.Call).get_source_segments(source)
    # for each candidate, disassemble and compare the instructions to what we
    # actually have, a match means this is the correct statement
    if not candidate_segments:
        raise RuntimeError('No calls found.')
    # disassemble and get the instructions up to the current position
    target_instrs = _instructions_up_to_offset(caller_frame.f_code,
                                               caller_frame.f_lasti)

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


def get_attribute_segment_from_frame(caller_frame: types.FrameType) -> str:
    # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
    # , for python 3.11 (currently in development) this can be done via co_positions
    # (see https://www.python.org/dev/peps/pep-0657/),
    # for now, as 3.11 isn't widely used, this requires the following hack:
    # extract the source of the class init statement
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = get_source(caller_frame.f_code.co_filename)
    source = get_statement(full_source, caller_frame.f_lineno)
    # the source can contain surrounding stuff we need to discard
    # as we only have the line number (this is what makes this complicated)

    # get all segments in the source that correspond to calls and might thus
    # potentially be the class init
    candidate_segments = var2mod.Finder(ast.Attribute).get_source_segments(source)
    # for each candidate, disassemble and compare the instructions to what we
    # actually have, a match means this is the correct statement
    if not candidate_segments:
        raise RuntimeError('No attributes found.')
    # disassemble and get the instructions up to the current position
    target_instrs = _instructions_up_to_offset(caller_frame.f_code,
                                               caller_frame.f_lasti)

    segment = None
    found = False

    for segment in sorted(candidate_segments, key=lambda x: -len(x)):

        instrs = _instructions_in_name(segment)
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
            found = True
            break

    if segment is None or not found:
        raise RuntimeError('Attribute not found.')

    return segment
