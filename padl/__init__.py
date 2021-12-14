from padl.wrap import transform
from padl.transforms import (
    Batchify, Unbatchify, identity, Identity, group, load, save, fulldump,
    importdump
)
from padl.dumptools.serialize import value
from padl.util_transforms import IfTrain, IfEval, IfInfer, IfInMode
from padl.version import __version__

#: this is same
from padl.utils import same

#: The *identity* Transform: *f(x) = x*.
identity = Identity()

Input = Identity() - 'InputNode'
Output = Identity() - 'OutputNode'

#: See :class:`Batchify`.
batch = Batchify()

#: See :class:`Unbatchify`.
unbatch = Unbatchify()

__all__ = ['value', 'transform', 'Batchify', 'Unbatchify', 'Identity', 'group', 'load', 'save',
           'IfTrain', 'IfEval', 'IfInfer', 'IfInMode', 'identity', 'batch', 'unbatch', 'same',
           '__version__', 'fulldump', 'importdump']
