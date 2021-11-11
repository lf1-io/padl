from padl.wrap import transform
from padl.transforms import Batchify, Unbatchify, Identity, group, load, save
from padl.dumptools.serialize import value
from padl.util_transforms import IfTrain, IfEval, IfInfer, IfInStage
from padl.version import __version__

#: this is this
from padl.utils import this

#: The *identity* Transform: *f(x) = x*.
identity = Identity()

#: See :class:`Batchify`.
batch = Batchify()

#: See :class:`Unbatchify`.
unbatch = Unbatchify()

__all__ = ['value', 'transform', 'Batchify', 'Unbatchify', 'Identity', 'group', 'load', 'save',
           'IfTrain', 'IfEval', 'IfInfer', 'IfInStage', 'identity', 'batch', 'unbatch', 'this',
           '__version__']
