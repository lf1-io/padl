import ast
from dataclasses import dataclass
import dis
import functools
import inspect
import linecache
from pathlib import Path
import sys
import types
from typing import List, Optional, Callable

import torch

from lf import var2mod, thingfinder


# TODO: remove (debug)
from IPython import embed


def _trace_this(tracefunc: Callable, frame: Optional[types.FrameType] = None):
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


def _get_source(filename):
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


def _get_statement(source, lineno):
    """Get complete (potentially multi-line) statement at line *lineno*. """
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


def _caller_frame(depth):
    """Get the caller frame in depth *depth*. """
    return inspect.stack()[depth + 1].frame


def _caller_module(depth):
    """Get the caller module in depth *depth*. """
    return _module(_caller_frame(depth + 1))


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


def _non_lf_caller():
    stack = inspect.stack()
    for frameinfo in stack:
        if frameinfo.frame.f_globals['__name__'] == __name__:
            continue
        return frameinfo


def _get_call_segment_from_frame(caller_frame):
    # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
    # , for python 3.11 (currently in development) this can be done via co_positions
    # (see https://www.python.org/dev/peps/pep-0657/),
    # for now, as 3.11 isn't widely used, this requires the following hack:
    # extract the source of the class init statement
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = _get_source(caller_frame.f_code.co_filename)
    source = _get_statement(full_source,
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


@dataclass
class _CallInfo:
    module: types.ModuleType
    function: Optional[str] = None
    scope: Optional[list] = None
    fdef_source: Optional[str] = None


def _set_local_varname(frame, event, _args):
    if event == 'return':
        for k, v in frame.f_locals.items():
            try:
                if v._lf_varname is _notset or v._lf_varname is None:
                    v.set_lf_varname(k)
            except AttributeError:
                continue


def _caller_info(depth, drop_n=0):
    stack = inspect.stack()
    caller = stack[depth + 1]
    call_info = _CallInfo(_module(caller.frame))
    call_info.function = caller.function
    if caller.function != '<module>':
        call_source = _get_call_segment_from_frame(stack[depth + 2].frame)
        call_info.fdef_source = _get_source(stack[depth + 1].filename)
        fdef_lineno = caller.frame.f_lineno
        call_info.scope = thingfinder.Scope.from_source(call_info.fdef_source, fdef_lineno,
                                                        call_source, call_info.module, drop_n)
    return call_info


def _wrap_function(fun):
    """Fram *fun* in a Transform. Don't use directly, use `trans` instead. """
    call_info = _caller_info(2, 1)
    caller = inspect.stack()[2]
    if call_info.function != '<module>':
        _trace_this(_set_local_varname, caller.frame)
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
            _get_call_segment_from_frame(_caller_frame(1)),
            _caller_info(1)
        )

    cls.__init__ = __init__
    return cls


def _wrap_lambda(fun):
    """Wrap a lambda function in a transform. Hacky hack that will hopefully
    become obsolete with python 3.11 (see _wrap_class). """
    # get the caller frame (it's 2 - [caller] -> [trans] -> [_wrap_lambda])
    caller_frame = _caller_frame(2)
    # get the source
    try:
        full_source = caller_frame.f_globals['_lf_source']
    except KeyError:
        full_source = _get_source(caller_frame.f_code.co_filename)
    source = _get_statement(full_source,
                            caller_frame.f_lineno)
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

    wrapper = FunctionTransform(fun, _CallInfo(_module(caller_frame)), call)
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


def build_codegraph(x, stack, res):
    todo = set(x)
    while todo and (next_ := todo.pop()):
        if next_ in res:
            continue
        source, node, scope = var2mod.thingfinder.find_in_stack(next_, stack)
        globals_ = var2mod.find_globals(node)
        determine_scope(globals_)
        res[next_] = (source, globals_, node)
        todo.update(globals_)
    return res


class _Notset:
    pass


_notset = _Notset()


class Transform:
    def __init__(self, call_info):
        self._lf_call_info = call_info
        self._lf_varname = _notset

    def __rshift__(self, other):
        return Compose([self, other], _caller_info(1), flatten=True)

    def __add__(self, other: "Transform") -> "Rollout":
        """ Rollout with *other*. """
        return Rollout([self, other], _caller_info(1), flatten=True)

    def __truediv__(self, other: "Transform") -> "Parallel":
        """ Parallel with *other*. """
        return Parallel([self, other], _caller_info(1), flatten=True)

    def lf_pre_save(self, path, i):
        pass

    def lf_post_load(self, path, i):
        pass

    def lf_save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        for i, subtrans in enumerate(self.lf_all_transforms_with_globals()):
            subtrans.lf_pre_save(path, i)
        with open(path / 'transform.py', 'w') as f:
            f.write(self.lf_dumps())

    def lf_codegraph(self):
        var_transforms = {}
        for transform in self.lf_all_transforms():
            if False and transform is self:
                continue
            varname = transform.lf_varname()
            if varname is not None:
                var_transforms[transform] = varname
        call = self.lf_evaluable_repr(var_transforms=var_transforms)
        assignment = f'_lf_main = {call}'
        if call != self.lf_varname() and self.lf_varname() is not None:
            assignment = f'{self.lf_varname()} = {assignment}'
        dependencies = var2mod._VarFinder().find(ast.parse(assignment)).globals
        graph = build_codegraph(dependencies, self._lf_call_info)
        graph[f'_lf_main'] = (
            assignment,
            dependencies,
            ast.parse(assignment).body[0]
        )
        return graph

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        return NotImplemented

    def lf_all_transforms(self):
        return NotImplemented

    def lf_all_transforms_with_globals(self):
        res = self.lf_all_transforms()
        graph = self.lf_codegraph()
        all_globals = set()
        for v in graph.values():
            all_globals.update(v[1])
        for g in all_globals:
            gobj = self._lf_call_info.module.__dict__[g]
            if gobj is self:
                continue
            self._lf_all_transforms_with_globals(gobj, res)
        return res

    @staticmethod
    def _lf_all_transforms_with_globals(x, res):
        if isinstance(x, str):
            return
        if isinstance(x, Transform):
            for child_transform in x.lf_all_transforms():
                if child_transform not in res:
                    res.append(child_transform)
        try:
            for t in x:
                Transform._lf_all_transforms_with_globals(t, res)
        except TypeError:
            pass

    def lf_dumps(self):
        return var2mod.dumps_graph(self.lf_codegraph())

    def lf_repr(self, indent=0):
        varname = self.lf_varname()
        evaluable_repr = self.lf_evaluable_repr()
        if varname is None or varname == evaluable_repr:
            return f'{evaluable_repr}'
        return f'{evaluable_repr} [{varname}]'

    def __repr__(self):
        return self.lf_repr()

    def _lf_find_varname(self, scopedict):
        try:
            return [
                k for k, v in scopedict.items()
                if v is self and not k.startswith('_')
            ][0]
        except IndexError:
            return None

    def lf_varname(self, scopedict=None):
        """The transform's variable name. """
        if self._lf_varname is _notset:
            if scopedict is None:
                scopedict = self._lf_call_info.module.__dict__
            self._lf_varname = self._lf_find_varname(scopedict)
        return self._lf_varname

    def lf_set_varname(self, val):
        self._lf_varname = val


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms, in contrast to `MetaTransform`s. """

    def __init__(self, call, call_info):
        super().__init__(call_info)
        self._lf_call = call

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        if var_transforms is None:
            var_transforms = {}
        return var_transforms.get(self, self._lf_call)

    def lf_all_transforms(self):
        res = [self]
        for v in self.__dict__.values():
            if isinstance(v, Transform):
                for child_transform in v.all_transforms():
                    if child_transform not in res:
                        res.append(child_transform)
        return res



class FunctionTransform(AtomicTransform):
    def __init__(self, function, call_info, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call, call_info)
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class TorchModuleTransform(AtomicTransform):
    def lf_pre_save(self, path, i):
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('saving torch module to', checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def lf_post_load(self, path, i):
        path = Path(path)
        checkpoint_path = path / f'{path.stem}_{i}.pt'
        print('loading torch module from', checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))


class MetaTransform(Transform):
    """Abstract base class for meta-trasforms (transforms combining other transforms. """
    op = NotImplemented

    def __init__(self, transforms, call_info, flatten=True):
        super().__init__(call_info)
        if flatten:
            transforms = self._flatten_list(transforms)
        self.transforms = transforms

    def lf_evaluable_repr(self, indent=0, var_transforms=None):
        sub_reprs = [
            x.lf_evaluable_repr(indent + 4, var_transforms)
            for x in self.transforms
        ]
        return (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )

    def lf_repr(self, indent=0):
        sub_reprs = [
            x.lf_repr(indent + 4)
            for x in self.transforms
        ]
        res = (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )
        if self.lf_varname() is not None and self.lf_varname() is not _notset:
            res += f' [{self.lf_varname()}]'
        return res

    @classmethod
    def _flatten_list(cls, transform_list: List[Transform]):
        """Flatten *list_* such that members of *cls* are not nested.

        :param list_: List of transforms.
        """
        list_flat = []

        for transform in transform_list:
            if isinstance(transform, cls):
                caller_frame = _non_lf_caller().frame
                if transform.lf_varname(caller_frame.f_locals) is None:
                    list_flat += transform.transforms
                else:
                    list_flat.append(transform)
            else:
                list_flat.append(transform)

        return list_flat

    def lf_all_transforms(self):
        res = [self]
        for transform in self.transforms:
            for child_transform in transform.lf_all_transforms():
                if child_transform not in res:
                    res.append(child_transform)
        return res


class Compose(MetaTransform):
    op = '>>'


class Rollout(MetaTransform):
    op = '+'


class Parallel(MetaTransform):
    op = '/'


def save(transform: Transform, path):
    transform.lf_save(path)


def load(path):
    """Load transform (as saved with lf.save) from *path*. """
    path = Path(path)
    with open(path / 'transform.py') as f:
        source = f.read()
    module = types.ModuleType('lfload')
    module.__dict__.update({
        '_lf_source': source,
        '_lf_module': module,
        '__file__': str(path / 'transform.py')
    })
    exec(source, module.__dict__)
    transform = module._lf_main
    for i, subtrans in enumerate(transform.lf_all_transforms_with_globals()):
        subtrans.lf_post_load(path, i)
    return transform
