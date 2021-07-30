# pylint: disable=not-callable
"""Utilities for lf"""
from contextlib import contextmanager
import os
import numpy
import torch

from torch.utils.data import DataLoader
from lf.dataproc.data import SimpleIterator


def make_loader(samples, model, *args, **kwargs):
    """
    Create a dataloader using *samples* (a sequence) and model's transform part.
    """

    dataset = SimpleIterator(samples, model.trans)
    loader = DataLoader(dataset, *args, **kwargs)

    return loader


def convert(batch, to):
    """Convert to device if needed.

    :param batch: nested tuples over ground tensors
    :param to: {'cuda', 'cpu'}
    """
    if isinstance(batch, torch.Tensor):
        batch = batch.to(to)
    else:
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = convert(batch[i], to)
        batch = tuple(batch)
    return batch


@contextmanager
def eval(model):
    """
    Temporarily put a transform into eval mode.

    This is a contextmanager, use like this:

        with eval(model):
            # model now in eval mode
            ...
        # model is now back in its previous mode

    :param model: lf transform
    """
    before = model.stage
    try:
        model.eval()
        yield
    finally:
        model.stage = before


@contextmanager
def evalmodel(model):
    """
    Temporarily put a model into eval mode.

    This is a contextmanager, use like this:

        with evalmodel(model):
            # model now in eval mode
            ...
        # model is now back in its previous mode
    :param model: lf transform
    """
    before = model.training
    try:
        model.eval()
        yield
    finally:
        if before:
            model.train()


@contextmanager
def stage(model, stage_):
    """
    Temporarily put a transform into a given stage.

    This is a contextmanager, use like this:

        with stage(model, 'infer'):
            # model now in infer mode
            ...
        # model is now back in its previous mode
    """
    previous_stage = model.stage
    try:
        model.stage = stage_
        yield
    finally:
        model.stage = previous_stage


def batchget(args, i):
    """
    Get the *i*th element of a tensor
    or
    get a tuple of the *i*th elements of a tuple (or list) of tensors

    >>> t1 = torch.Tensor([1,2,3])
    >>> t2 = torch.Tensor([4,5,6])
    >>> batchget(t1, 1)
    tensor(2)
    >>> batchget((t1, t2), 1)
    (tensor(2), tensor(5))

    :param args: arguments
    :param i: index in batch
    """
    if isinstance(args, torch.Tensor):
        return args[i]
    if isinstance(args, list) or isinstance(args, tuple):
        return tuple([batchget(args[j], i) for j in range(len(args))])
    raise TypeError


def unbatch(args):
    """
    Convert an input in batch-form into a tuple of datapoints.

    E.g:
        ([1, 4], ([2, 5], [3, 6])) -> [(1, (2, 3)), (4, (5, 6))]

    :param args: arguments to be unbatched
    """
    out = []
    itr = 0
    while True:
        try:
            temp = batchget(args, itr)
            out.append(temp)
            itr += 1
        except IndexError:
            return out


def make_batch(items):
    """
    Take a tuple or list of data points and convert them into batches.

    E.g:
        [(1, (2, 3)), (4, (5, 6))] -> ([1, 4], ([2, 5], [3, 6]))

    :param items: nested tuples over tensors/ objects convetable to tensors
    """
    if isinstance(items[0], torch.Tensor):
        r = torch.stack(items)
        return r
    if isinstance(items[0], list) or isinstance(items[0], tuple):
        transposed = zip(*items)
        return [make_batch(t) for t in transposed]

    return torch.tensor(items)


def subsample(iterator, n=1000, seed=1):
    """
    Subsample *n* items of a sequence *iterator*.

    If `len(iterator) <= n`, return *iterator*.

    :param iterator: iterator object
    :param n: number to subsample
    :param seed: random seed
    """

    if len(iterator) <= n:
        return iterator

    random_state = numpy.random.get_state()
    numpy.random.seed(seed)
    idx = numpy.random.choice(range(n), replace=False, size=n)
    # reset state to what it was before
    numpy.random.set_state(random_state)

    class Subsample:
        """Iterator that that subsamples from the original iterator"""
        def __getitem__(self, item):
            return iterator[int(idx[item])]

        def __len__(self):
            return n

    return Subsample()


def this(file, checkpoints='checkpoints'):
    """
    Get the checkpoint name as a string using the config file path.

    Inside config_.py:
    >>> experiment_name = this(__file__)

    :param file: get the path of the current directory
    :param checkpoints: checkpoints prefix
    """
    experiment = os.path.abspath(file).split('{}/'.format(checkpoints))[1]
    experiment = '/'.join(experiment.split('/')[:-1])
    return experiment
