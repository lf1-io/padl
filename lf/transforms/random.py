# pylint: disable=arguments-differ
"""Transforms that use functionality with random components"""
import random
import time

import numpy

from lf.transforms.core import Transform, ListTransform, trans, Compose
from lf.typing import types as t


@trans(t.List(len='?a'), t.List(len='?a'))
def shuffle(l):
    """Shuffle the input list *l*"""
    perm = list(numpy.random.permutation(len(l)))
    return [l[i] for i in perm]


class Disjunction(ListTransform):
    """Choose a random transform from a list of transforms to apply to *args

    :param trans_list: list of transforms
    """
    # ??a todo: check types
    def __init__(self, trans_list, p=()):
        super().__init__(trans_list, p=p)

    def do(self, *args):
        a_function = numpy.random.choice([*self.trans_list, lambda x: x], p=self.p)
        return a_function(*args)


class Sample(Transform):
    """Sample *n* elements from a List."""

    def __init__(self, n, default, p=None):
        super().__init__(n=n, default=default, p=p,
                         in_type=t.Sequence(), out_type=t.List(len=n))

    def do(self, l_):
        if not l_:  # ??a move out of here, solve with an If transform
            return [self.default for _ in range(self.n)]

        if len(l_) <= self.n:  # ??a don't allow lists shorter than n
            random.shuffle(l_)
            return l_

        return list(
            numpy.random.choice(l_, p=self.p, size=self.n, replace=False)
        )


class RandomApply(Transform):
    """ Randomly apply transforms."""

    def __init__(self, transform, p=0.5):
        if isinstance(transform, (list, tuple)):
            raise ValueError('list and tuple not supported')
        super().__init__(transform, p=p)

    @classmethod
    def _alternate_init_(cls, trans_list, p=0.5):
        a_trans = Compose(trans_list)
        return cls(a_trans, p=p)

    def do(self, x):
        if random.random() < self.p:
            return self.transform.do(x)
        return x

    @property
    def trans(self):
        return type(self)(transform=self.transform.trans, **self.kwargs)

    def __repr__(self):
        return 'lf.transforms.RandomApply({})'.format(self.transform)


class RandomOrder(ListTransform):
    # ??a todo: check types, all need to have same output and input
    """ Apply a list of transforms in random order."""

    def __init__(self, trans_list):
        super().__init__(trans_list)

    def do(self, x):
        order = list(range(len(self.trans_list)))
        random.shuffle(order)
        for i in order:
            x = self.trans_list[i].do(x)
        return x

    def __repr__(self):
        start = 'lf.transforms.RandomOrder([\n'
        lines = []
        for a_trans in self.trans_list:
            lines.extend(['    ' + y for y in str(a_trans).split('\n')])
        body = '\n'.join(lines)
        end = '\n])'
        return start + body + end


class Synchronized(ListTransform):
    """Execute a tuple of random transforms with the same random seed, such that
    random events are synchronized."""

    def __init__(self, trans_list):
        in_type = tuple(t.type.x.copy() for t in trans_list)
        out_type = tuple(t.type.y.copy() for t in trans_list)
        super().__init__(trans_list=trans_list, in_type=in_type, out_type=out_type)

    def do(self, arg_list):
        seed = int(time.time() * 1000000 + random.randint(0, 10000000000))
        res = []
        for i, arg in enumerate(arg_list):
            random.seed(seed)
            res.append(self.trans_list[i].do(arg))
        return res
