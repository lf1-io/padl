import ast
import dis
import functools
import inspect
import linecache
import sys
import types
import torch
from pathlib import Path
from typing import Union
from typing import List
from collections import namedtuple, Counter

from lf import var2mod


# TODO: remove (debug)
from IPython import embed


def _isinstance_of_namedtuple(arg):
    """Check if input is instance of namedtuple"""
    typ = type(arg)
    base = typ.__bases__
    if len(base) != 1 or base[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(field, str) for field in fields)


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


def _wrap_function(fun):
    """Fram *fun* in a Transform. Don't use directly, use `trans` instead. """
    wrapper = FunctionTransform(fun, _caller_module(2), _same_module_stack(2))
    functools.update_wrapper(wrapper, fun)
    return wrapper


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
    res = []
    stack = inspect.stack()[depth + 1:]
    module_name = stack[0].frame.f_globals['__name__']
    for f in stack:
        if f.frame.f_globals['__name__'] == module_name:
            res.append(f.frame)
        else:
            break
    return res


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
        # we want to extract the precise init statement here (e.g. `MyClass(1, 2, 3)`
        # , for python 3.11 (currently in development) this can be done via co_positions
        # (see https://www.python.org/dev/peps/pep-0657/),
        # for now, as 3.11 isn't widely used, this requires the following hack:

        # get the caller frame
        caller_frame = _caller_frame(1)
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
            raise RuntimeError('No class initializations found.')

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
        else:
            raise RuntimeError('Class initialization not found.')

        old__init__(self, *args, **kwargs)
        trans_class.__init__(self, segment, _module(caller_frame), _same_module_stack(1))

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
    for segment, call in zip(candidate_segments, candidate_calls):
        instrs = list(dis.get_instructions(eval(segment)))
        if not len(instrs) == len(target_instrs):
            continue
        for instr, target_instr in zip(instrs, target_instrs):
            if (instr.opname, target_instr.argval) != (instr.opname, target_instr.argval):
                break
        else:
            break
    else:
        raise RuntimeError('Lambda not found.')

    wrapper = FunctionTransform(fun, _module(caller_frame), _same_module_stack(2), call)
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


class Transform:
    def __init__(self, module, stack, lf_name=None):
        self._lf_module = module
        self._lf_stack = stack
        self.__lf_name = lf_name

    @property
    def lf_name(self):
        if self.__lf_name is None:
            return self.lf_varname
        return self.__lf_name

    def __rshift__(self, other):
        return Compose([self, other], _caller_module(1), flatten=True)

    def __add__(self, other: "Transform") -> "Rollout":
        """ Rollout with *other*. """
        return Rollout([self, other], _caller_module(1), flatten=True)

    def __truediv__(self, other: "Transform") -> "Parallel":
        """ Parallel with *other*. """
        return Parallel([self, other], _caller_module(1), flatten=True)

    def __sub__(self, transform_name: str) -> "Transform":
        """Name Transform"""
        return self.lf_clone(lf_name=transform_name)

    def lf_clone(self, **kwargs):
        """Clone Transform"""
        return NotImplementedError

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
            varname = transform.lf_varname
            if varname is not None:
                var_transforms[transform] = varname
        call = self.lf_evaluable_repr(var_transforms=var_transforms)
        assignment = f'_lf_main = {call}'
        if call != self.lf_varname and self.lf_varname is not None:
            assignment = f'{self.lf_varname} = {assignment}'
        dependencies = var2mod._VarFinder().find(ast.parse(assignment)).globals
        graph = var2mod.build_codegraph(dependencies, self._lf_module)
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
            gobj = self._lf_module.__dict__[g]
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
        varname = self.lf_varname
        evaluable_repr = self.lf_evaluable_repr()
        if varname is None or varname == evaluable_repr:
            return f'{evaluable_repr}'
        return f'{evaluable_repr} [{varname}]'

    def __repr__(self):
        return self.lf_repr()

    @property
    def lf_varname(self):
        """The transform's variable name. """
        try:
            return [
                k for k, v in self._lf_module.__dict__.items()
                if v is self and not k.startswith('_')
            ][0]
        except IndexError:
            return None

    def __call__(self, arg):
        return NotImplementedError

    def _lf_call_transform(self, arg):
        """Call transform with possiblity to pass multiple arguments"""
        signature_parameters = inspect.signature(self).parameters
        if len(signature_parameters) == 1:
            return self(arg)
        return self(*arg)


class AtomicTransform(Transform):
    """Base class for "atomic" transforms (transforms that are not made by combining
    other transforms, in contrast to `CompoundTransform`s. """

    def __init__(self, call, module, stack):
        super().__init__(module, stack)
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
    def __init__(self, function, module, stack, call=None):
        if call is None:
            call = function.__name__
        super().__init__(call, module, stack)
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


class CompoundTransform(Transform):
    """Abstract base class for compound-transforms (transforms combining other transforms.)"""
    op = NotImplemented

    def __init__(self, transforms, module, stack, flatten=True, lf_name=None, **kwargs):

        super().__init__(module, stack, lf_name=lf_name)

        self.__lf_group = kwargs.get('__lf_group', False) if lf_name is None else True
        #assert not (self.lf_name is not None and self.__lf_group is False)

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
        return (
            '(\n    ' + ' ' * indent
            + ('\n' + ' ' * indent + f'    {self.op} ').join(sub_reprs)
            + '\n' + ' ' * indent + ')'
        )

    @classmethod
    def _flatten_list(cls, transform_list: List[Transform]):
        """Flatten *list_* such that members of *cls* are not nested.

        :param list_: List of transforms.
        """
        list_flat = []

        for transform in transform_list:
            if isinstance(transform, cls):
                if transform.__lf_group:
                    list_flat.append(transform)
                else:
                    list_flat += transform.transforms
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

    @classmethod
    def return_grouped(cls, transform):
        return cls(transform.transforms, transform.module, transform.stack,
                   transform.flatten, transform.lf_name, __lf_group=True)

    @staticmethod
    def _lf_get_keys(transforms):
        """Get deduplicated keys from list of transforms

        Names are updated as below.
        [None, None, 'a', 'a', 'b', 'c'] -> ['out_0', 'out_1', 'a_0', 'a_1', 'b', 'c']

        :param transforms: list of transforms
        :return: list of keys
        """
        names = []
        for ind, transform_ in enumerate(transforms):
            if transform_.lf_name is None:
                name = 'out_'+str(ind)
            else:
                name = transform_.lf_name
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
    :param flatten: If *True* flatten transforms -
        Compose([Compose([a,b]), c]) becomes Compose([a, b, c])
    :return: output from series of transforms
    """
    op = '>>'

    def __call__(self, arg):
        """Call method for Compose

        :param arg: arguments to call with
        :return: output from series of transforms
        """
        for transform_ in self.transforms:
            arg = transform_._lf_call_transform(arg)
        return arg


class Rollout(CompoundTransform):
    """Apply a list of transform to same input and get tuple output

    Rollout([t1, t2, ...])(x) := (t1(x), t2(x), ...)

    :param transforms: List of transforms to rollout.
    :param flatten: If *True* flatten transforms -
        Rollout([Rollout([a,b]), c]) becomes Rollout([a, b, c])
    :return: namedtuple of outputs
    """
    op = '+'

    def __init__(self, transforms, module, stack, flatten=False, lf_name=None, **kwargs):
        keys = self._lf_get_keys(transforms)
        super().__init__(transforms, module, stack, flatten=flatten, lf_name=lf_name, **kwargs)
        self.lf_keys = keys
        self._lf_output_format = namedtuple('namedtuple', self.lf_keys)

    def __call__(self, arg):
        """Call method for Rollout

        :param arg: Argument to call with
        :return: namedtuple of outputs
        """
        out = []
        for transform_ in self.transforms:
            out.append(transform_._lf_call_transform(arg))
        out = self._lf_output_format(*out)
        return out


class Parallel(CompoundTransform):
    """Apply transforms in parallel to a tuple of inputs and get tuple output

    Parallel([f1, f2, ...])((x1, x2, ..)) := (f1(x1), f2(x2), ...)

    :param transforms: List of transforms to parallelize.
    :param flatten: If *True* flatten transforms -
        Parallel([Parallel([a,b]), c]) becomes Parallel([a, b, c])
    :return: namedtuple of outputs
    """
    op = '/'

    def __init__(self, transforms, module, stack, flatten=False, lf_name=None, **kwargs):
        keys = self._lf_get_keys(transforms)
        super().__init__(transforms, module, stack, flatten=flatten, lf_name=lf_name, **kwargs)
        self.lf_keys = keys
        self._lf_output_format = namedtuple('namedtuple', self.lf_keys)

    def __call__(self, arg):
        """Call method for Parallel

        :param arg: Argument to call with
        :return: namedtuple of output
        """
        out = []
        for ind, transform_ in enumerate(self.transforms):
            out.append(transform_._lf_call_transform(arg[ind]))
        out = self._lf_output_format(*out)
        return out


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


def group(transform: Union[Rollout, Parallel]):
    return type(transform).return_grouped(transform)
