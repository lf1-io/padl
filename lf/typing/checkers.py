"""Typing checkers"""
from .types import Mismatch, convert, free


class JitError(Exception):
    ...


def check_python_type(types_):
    """Check the python type"""
    def wrapper(x, y):
        failed = 0
        for a_type in types_:
            if isinstance(x, a_type):
                continue
            else:
                failed += 1
        if failed == len(types_):
            str_types = " or ".join([str(t_) for t_ in types_])
            raise JitError(
                f'transform type needs {str_types}; got {type(x)}'
            )
    return wrapper


class IterableChecker:
    def __init__(self, f):
        self.f = f

    def __call__(self, x, y):

        self.f(x, None)

        rval = y.copy({}).rval
        if 'len' in rval:
            try:
                rval['len'] @ len(x)
            except Mismatch:
                raise JitError(
                    f'type expected length {rval["len"]}; '
                    f'got run-time len {len(x)}'
                )

        if 'vtype' in rval:
            for i, z in enumerate(x):
                if free(rval['vtype']):
                    continue
                try:
                    rval['vtype'].jit_check(z)
                except JitError as err:
                    raise JitError(
                        f'jit type-mismatch at position {i}; ' + str(err)
                    )


class ShapeChecker:
    def __init__(self, base_type, callback=lambda x: x):
        self.type = base_type
        self.callback = callback

    def __call__(self, x, y):
        if not isinstance(self.callback(x), self.type):
            raise JitError(
                f'expected type {self.type}; got {type(x)}'
            )
        shape = y.copy({}).rval['shape']
        try:
            shape @ convert(list(x.shape))
        except Mismatch:
            raise JitError(
                f'type expected {shape}; '
                f'got run-time shape {x.shape}'
            )
