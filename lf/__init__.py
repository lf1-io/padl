from lf.wrap import trans

from lf.transform import Batchify, Unbatchify, Identity, group
identity = Identity()
batch = Batchify()
unbatch = Unbatchify()
from lf.utils import this
