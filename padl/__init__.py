from padl.wrap import transform
from padl.transforms import (
    Batchify, Unbatchify, identity, Identity, group, load, save, fulldump,
    importdump
)
from padl.dumptools.serialize import value, param
from padl.util_transforms import IfTrain, IfEval, IfInfer, IfInMode
from padl.utils import pd_debug
from padl.version import __version__

#: this is same
from padl.utils import same
batch = Batchify()

#: See :class:`Unbatchify`.
unbatch = Unbatchify()

__all__ = ['value', 'transform', 'Batchify', 'Unbatchify', 'Identity', 'group', 'load', 'save',
           'IfTrain', 'IfEval', 'IfInfer', 'IfInMode', 'identity', 'batch', 'unbatch', 'same',
           '__version__', 'fulldump', 'importdump', 'param', 'pd_debug']

try:
    import inspect
    from padl.dumptools.sourceget import get_source
    _ = get_source(inspect.stack()[-1].filename)
    # We do not want these to be padl level imports so we remove them
    del inspect
    del get_source
except Exception as e:
    raise RuntimeError('PADL does not work in the current Interpreter Environment because we '
                       'rely on the inspect module to find source code. '
                       'Unfortunately, the source code typed at this interactive '
                       'prompt is discarded as soon as it is parsed. Therefore, we recommend '
                       'using the IPython interpreter or Jupyter Notebooks for interactive '
                       'sessions.') from e
