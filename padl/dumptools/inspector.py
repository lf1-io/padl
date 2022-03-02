"""Utilities for inspecting frames. """

import ast
import dis
import inspect
import sys
import types
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Callable, Optional
from warnings import warn

from padl.dumptools import ast_utils, symfinder, var2mod
from padl.dumptools.sourceget import get_source, original, cut


class CallInfo:
    """Information about the calling context.

    Contains the following information:
        - the function from which that call was made
        - the scope (see :class:`symfinder.Scope`)

    :param origin: Where to look for the call, can be
        - "nextmodule": use the first frame not in the module the object was created in
        - "here": use the frame the object was created in
    :param drop_n: Drop *n* levels from the calling scope.
    :param ignore_scope: Don't try to determine the scope (use the toplevel scope instead).
    """

    def __init__(self, origin: Literal['nextmodule', 'here'] = 'nextmodule',
                 drop_n: int = 0, ignore_scope: bool = False):
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
                         drop_n: int, ignore_scope: bool) -> symfinder.Scope:
        """Determine the scope of the caller frame. """
        module = _module(caller_frameinfo.frame)

        if self.function == '<module>' or ignore_scope:
            return symfinder.Scope.toplevel(module)

        return _get_scope_from_frame(caller_frameinfo.frame, drop_n)

    @property
    def module(self):
        """The calling module. """
        return self.scope.module


# exclude these modules from detailed scope analysis (as that slows testing down in python 3.7.)
_EXCLUDED_MODULES = ['pytest', 'pluggy']


def _get_scope_from_frame(frame, drop_n):
    """Get the :class:`~symfinder.Scope` from the frame object *frame*. """
    module = _module(frame)
    # don't dive deeper for excluded modules
    if any(module.__name__.startswith(excluded_module) for excluded_module in _EXCLUDED_MODULES):
        return symfinder.Scope.toplevel(module)
    # don't dive deeper after
    if module.__name__ == '__main__' and _module(frame.f_back) != module:
        return symfinder.Scope.toplevel(module)
    # don't dive deeper if not a call
    if frame.f_code.co_name == '<module>':

        return symfinder.Scope.toplevel(module)
    try:
        call_source = get_segment_from_frame(frame.f_back, 'call')
    except (RuntimeError, FileNotFoundError, AttributeError, OSError):
        return symfinder.Scope.toplevel(module)
    try:
        definition_source = get_source(frame.f_code.co_filename)
    except FileNotFoundError:
        return symfinder.Scope.toplevel(module)
    calling_scope = _get_scope_from_frame(frame.f_back, 0)
    scope = symfinder.Scope.from_source(definition_source, frame.f_lineno,
                                        call_source, module, drop_n,
                                        calling_scope)
    assert len(scope) <= 1, 'scope longer than 1 currently not supported'
    return scope


def non_init_caller_frameinfo() -> inspect.FrameInfo:
    """Get the FrameInfo for the first outer frame that is not of an "__init__" method. """
    stack = inspect.stack()
    frameinfo = None
    for frameinfo in stack[1:]:
        if frameinfo.function != '__init__':
            break
    assert frameinfo is not None and frameinfo.function != '__init__'
    return frameinfo


def trace_this(tracefunc: Callable, frame: Optional[types.FrameType] = None, *args, **kwargs):
    """Call in a function body to trace the rest of the function execution with function
    *tracefunc*. *tracefunc* must match the requirements for the argument of `sys.settrace`
    (in the documentation of which more details can be found).

    Example:

    >>> def tracefunc(frame, event, arg):
    ...     if 'event' == 'return':
    ...         print('returning', arg)

    >>> def myfunction():
    ...     [...]
    ...     _trace_this(tracefunc)
    ...     return 123

    :param tracefunc: Trace function (see documentation of `sys.settrace` for details).
    :param frame: The frame to trace (defaults to the caller's frame).
    """
    previous_tracefunc = sys.gettrace()

    if frame is None:
        # default is the caller's frame
        frame = inspect.currentframe().f_back

    def trace(frame, event, arg):
        tracefunc(frame, event, arg, *args, **kwargs)
        if event == 'return':
            sys.settrace(previous_tracefunc)
        if previous_tracefunc is not None:
            previous_tracefunc(frame, event, arg)

    if previous_tracefunc is None:
        # set global tracefunc to something, this is required to enable local tracing
        sys.settrace(lambda _a, _b, _c: None)

    frame.f_trace = trace


def _instructions_up_to_call(x) -> list:
    """Get all instructions up to last CALL FUNCTION. """
    instructions = list(dis.get_instructions(x))
    i = 0
    for i, instruction in enumerate(instructions[::-1]):
        if instruction.opname.startswith('CALL_'):
            break
    return instructions[:-i]


def _instructions_in_name(x) -> list:
    """Get all LOAD_NAME and LOAD_ATTR instructions. """
    instructions = list(dis.get_instructions(x))
    i = 0
    for i, instruction in enumerate(instructions):
        if instruction.opname not in ('LOAD_NAME', 'LOAD_ATTR'):
            break
    return instructions[:i]


def _instructions_in_getitem(x) -> list:
    """Get all instructions up to last CALL FUNCTION. """
    instructions = list(dis.get_instructions(x))
    return instructions[:-1]


def _instructions_up_to_offset(x, lasti: int) -> list:
    """Get all instructions up to offset *lasti*. """
    instructions = []
    for instruction in dis.get_instructions(x):
        instructions.append(instruction)
        if instruction.offset == lasti:
            break
    return instructions


def get_statement(source: str, lineno: int):
    """Get complete (potentially multi-line) statement at line *lineno* out of *source*.

    :returns: A tuple of statement and offset. The offset is a tuple of row offset and col offset.
        It can be used to determine the location of the statement within the source.
    """
    for row_offset in range(lineno):
        try:
            block, lineno_in_block, col_offset = get_surrounding_block(source, lineno - row_offset)
        except ValueError:
            continue
        try:
            try:
                statement, offset = _get_statement_from_block(block, lineno_in_block + row_offset)
                return statement, (lineno - offset - 1, -col_offset)
            except SyntaxError:
                statement, offset = _get_statement_from_block('(\n' + block + '\n)',
                                                              lineno_in_block + row_offset + 1)
                return statement, (lineno - lineno_in_block - 1, -col_offset)
        except SyntaxError:
            continue
    raise SyntaxError("Couldn't find the statement.")


def _get_statement_from_block(block: str, lineno_in_block: int):
    """Get statements at like *lineno_in_block* from a code block.

    :param block: The code block to get the statements from.
    :param lineno_in_block: Line number which the statement must include. Counted from 1.
    :returns: A tuple of string with the found statements and and offset between the beginning of
        the match and *lineno_in_block*.
    """
    module = ast.parse(block)
    stmts = []
    offset = None
    for stmt in module.body:
        position = ast_utils.get_position(block, stmt)
        if position.lineno <= lineno_in_block <= position.end_lineno:
            stmts.append(ast_utils.get_source_segment(block, stmt))
            if offset is None:
                offset = lineno_in_block - position.lineno
    if offset is None:
        offset = 0
    return '\n'.join(stmts), offset


def get_surrounding_block(source: str, lineno: int):
    """Get the code block surrounding the line at *lineno* in *source*.

    The code block surrounding a line is the largest block of lines with the same or larger
    indentation as the line itself.

    The result will be unindented.

    Raises a `ValueError` if the line at *lineno* is empty.

    :param source: The source to extract the block from.
    :param lineno: Number of the line for extracting the block.
    :returns: A tuple containing the block itself and the line number of the target line
        within the block and the number of spaces removed from the front.
    """
    lines = source.split('\n')
    before, after = lines[:lineno-1], lines[lineno:]
    white = _count_leading_whitespace(lines[lineno-1])
    if white is None:
        raise ValueError('Line is empty.')
    block = [lines[lineno-1][white:]]
    lineno_in_block = 1

    # get all lines of the same block before *lineno*
    while before:
        next_ = before.pop(-1)
        next_white = _count_leading_whitespace(next_)
        starts_with_comment = next_.lstrip().startswith('#')
        if next_.strip().endswith(':'):
            break
        if next_white is None or next_white >= white or starts_with_comment:
            block = [next_[white:]] + block
        else:
            break
        lineno_in_block += 1

    # remove leading rows with more than *white* leading whitespace
    # (e.g. hanging function arguments)
    while block:
        next_white = _count_leading_whitespace(block[0])
        if next_white is None or next_white == 0:
            break
        block.pop(0)
        lineno_in_block -= 1

    # get all lines of the same block after *lineno*
    while after:
        next_ = after.pop(0)
        next_white = _count_leading_whitespace(next_)
        if next_white is None or next_white >= white:
            block = block + [next_[white:]]
        else:
            break
    return '\n'.join(block), lineno_in_block, white


def _module(frame: types.FrameType) -> types.ModuleType:
    """Get module of *frame*. """
    try:
        return frame.f_globals['_pd_module']
    except KeyError:
        return sys.modules[frame.f_globals['__name__']]


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


def get_segment_from_frame(caller_frame: types.FrameType, segment_type, return_locs=False) -> str:
    """Get a segment of a given type from a frame.

    *NOTE*: All this is rather hacky and should be changed as soon as python 3.11 becomes widely
    available as then it will be possible to get column information from frames
    (see inline comments).

    *segment_type* can be 'call', 'attribute', 'getitem'.
    """
    if segment_type == 'call':
        node_type = ast.Call
        instructions_finder = _instructions_up_to_call
    elif segment_type == 'attribute':
        node_type = ast.Attribute
        instructions_finder = _instructions_in_name
    elif segment_type == 'getitem':
        node_type = ast.Subscript
        instructions_finder = _instructions_in_getitem
    # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
    # , for python 3.11 (currently in development) this can be done via co_positions
    # (see https://www.python.org/dev/peps/pep-0657/),
    # for now, as 3.11 isn't widely used, this requires the following hack:
    # extract the source of the class init statement
    try:
        full_source = caller_frame.f_globals['_pd_source']
    except KeyError:
        full_source = get_source(caller_frame.f_code.co_filename)

    lineno = caller_frame.f_lineno
    # in python <= 3.7, the lineno points to the end of
    # the statement rather than the beginning, therefore we need to decrement it
    if segment_type == 'call':
        lines = original(full_source).split('\n')
        while '(' not in lines[lineno - 1] and not lines[lineno - 1].startswith('@'):
            lineno -= 1

    source, offset = get_statement(original(full_source), lineno)
    # the source can contain surrounding stuff we need to discard
    # as we only have the line number (this is what makes this complicated)

    # get all segments in the source that correspond to calls and might thus
    # potentially be the class init
    candidate_segments = var2mod.Finder(node_type).get_source_segments(source)
    # for each candidate, disassemble and compare the instructions to what we
    # actually have, a match means this is the correct statement
    if not candidate_segments:
        raise RuntimeError(f'{segment_type} not found.')
    # disassemble and get the instructions up to the current position
    target_instrs = _instructions_up_to_offset(caller_frame.f_code,
                                               caller_frame.f_lasti)

    # filter out EXTENDED_ARG (instruction used for very large args in target_instrs, this won't
    # be present in instrs)
    target_instrs = [x for x in target_instrs if x.opname != 'EXTENDED_ARG']

    segment = None
    locs = None
    found = False

    for segment, locs in sorted(candidate_segments, key=lambda x: -len(x[0])):

        instrs = instructions_finder(segment)
        if len(instrs) > len(target_instrs):
            continue
        for instr, target_instr in zip(instrs, target_instrs[-len(instrs):]):
            if (target_instr.opname == 'LOAD_FAST'
                    and target_instr.argval in caller_frame.f_locals
                    and target_instr.argval in caller_frame.f_code.co_varnames):
                argval = caller_frame.f_locals[target_instr.argval]
            else:
                argval = target_instr.argval
            instr_argval = getattr(instr.argval, 'co_code', instr.argval)
            target_argval = getattr(target_instr.argval, 'co_code', target_instr.argval)
            if instr_argval not in (target_argval, argval):
                break
            same_opname = instr.opname == target_instr.opname
            load_ops = ('LOAD_NAME', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_CONST', 'LOAD_DEREF')
            both_load = instr.opname in load_ops and target_instr.opname in load_ops
            if not (same_opname or both_load):
                break
        else:
            found = True
            break

    if segment is None or not found:
        raise RuntimeError(f'{segment_type} not found.')

    locs = (
        locs.lineno - 1 + offset[0],
        locs.end_lineno - 1 + offset[0],
        locs.col_offset - offset[1],
        locs.end_col_offset - offset[1]
    )
    # cutting is necessary instead of just using the segment from above for support of
    # `sourceget.ReplaceString`s
    segment = cut(full_source, *locs)

    if return_locs:
        return (
            segment, locs
        )
    return segment


def _count_leading_whitespace(line: str) -> int:
    """Count the number of spaces *line* starts with. If *line* is empty, return *None*."""
    i = 0
    for char in line:
        if char == ' ':
            i += 1
            continue
        return i
    return None
