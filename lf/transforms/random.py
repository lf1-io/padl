# pylint: disable=arguments-differ
"""Transforms that use functionality with random components"""
import random
import time
from warnings import warn

import numpy

from lf.transforms.core import Transform, ListTransform, trans, WrapperTransform, Compose
from lf.typing import types as t


class ShuffleList(Transform):
    """
    Shuffle list of objects
    ??a: make function transform
    """

    def __init__(self):
        warn('Deprecated: Use "shuffle" instead of "ShuffleList()"')
        super().__init__(in_type=t.List(len='?a'), out_type=t.List(len='?a'))

    def do(self, l_):
        perm = list(numpy.random.permutation(len(l_)))
        return [l_[i] for i in perm]


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


class RandomApply(WrapperTransform):
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


class ClipInput(Transform):
    """Clip the input"""

    def __init__(self, n=95):
        super().__init__(n=n)
        self._p = self.n - numpy.arange(self.n)
        self._p = self._p / self._p.sum()

    def do(self, input_):
        if self._stage != 'train':
            return input_

        new_len = int(numpy.random.multinomial(1, self._p).nonzero()[0][0] + 5)
        new_len = min(new_len, len(input_))

        if new_len < len(input_):
            start = numpy.random.randint(0, len(input_) - new_len)
            return input_[start:start + new_len]

        return input_


class ClipInputBatch(Transform):
    """Clip the input batch"""
    def __init__(self, n=95):
        super().__init__(n=n)
        self._p = self.n - numpy.arange(self.n)
        self._p = self._p / self._p.sum()

    def do(self, x):
        input_ = x[0]
        lens = x[1]
        new_len = int(numpy.random.multinomial(1, self._p).nonzero()[0][0] + 5)
        new_len = min(new_len, max(lens).item())
        for i, a_len in enumerate(lens):
            if a_len.item() <= new_len:
                continue
            start = numpy.random.randint(a_len.item() - new_len)
            lens[i] = new_len
            input_[i, :a_len] = input_[i, start:start+a_len].clone()
        return input_[:, :new_len], lens


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
