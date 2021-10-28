from lf.wrap import trans
from lf.transform import Batchify, Unbatchify, Identity, group
from lf.util_transforms import IfTrain, IfEval, IfInfer, IfInStage
identity = Identity()
batch = Batchify()
unbatch = Unbatchify()
from lf.utils import this