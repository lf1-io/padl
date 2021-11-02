"""Extra useful Transforms. """

from collections import OrderedDict
from typing import Optional

from padl.transforms import ClassTransform, Identity, Transform, Stage


class IfInStage(ClassTransform):
    """Perform *if_* if called in stage *target_stage*, else perform *else_*.

    Example:

        >>> a = transform(lambda x: x + 10)
        >>> b = transform(lambda x: x * 10)
        >>> iis = IfInStage(a, 'infer', b)
        >>> iis.infer_apply(1)
        11
        >>> list(iis.eval_apply([1]))
        [100]

    :param if_: Transform to apply when the stage matches.
    :param target_stage: Stage (one of 'train', 'eval', 'infer').
    :param else_: Transform to apply when the stage doesn't match (defaults to identity transform).
    """

    pd_dont_dump_code = True

    def __init__(self, if_: Transform, target_stage: Stage, else_: Optional[Transform] = None):
        super().__init__(arguments=OrderedDict([('if_', if_),
                                                ('target_stage', target_stage),
                                                ('else_', else_)]))

        assert target_stage in ('train', 'eval', 'infer'), "Target stage can only be train, " \
                                                           "eval or infer"

        if else_ is None:
            else_ = Identity()

        self.if_ = if_
        self.else_ = else_
        self.target_stage = target_stage
        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

        self._pd_component = set.union(*[t.pd_component for t in [self.if_, self.else_]])

    def __call__(self, args):
        assert Transform.pd_stage is not None, ('Stage is not set, use infer_apply, eval_apply '
                                                'or train_apply instead of calling the transform '
                                                'directly.')

        if Transform.pd_stage == self.target_stage:
            return self.if_._pd_call_transform(args)
        return self.else_._pd_call_transform(args)

    @property
    def pd_preprocess(self):
        if self._pd_preprocess is None:
            self._pd_preprocess = type(self)(
                if_=self.if_.pd_preprocess,
                target_stage=self.target_stage,
                else_=self.else_.pd_preprocess,
            )
        return self._pd_preprocess

    @property
    def pd_forward(self):
        if self._pd_forward is None:
            self._pd_forward = type(self)(
                if_=self.if_.pd_forward,
                target_stage=self.target_stage,
                else_=self.else_.pd_forward,
            )
        return self._pd_forward

    @property
    def pd_postprocess(self):
        if self._pd_postprocess is None:
            self._pd_postprocess = type(self)(
                if_=self.if_.pd_postprocess,
                target_stage=self.target_stage,
                else_=self.else_.pd_postprocess,
            )
        return self._pd_postprocess


class IfInfer(IfInStage):
    """Perform *if_* if called in "infer" stage, else perform *else_*.

    :param if_: Transform for the "infer" stage.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'infer', else_)


class IfEval(IfInStage):
    """Perform *if_* if called in "eval" stage, else perform *else_*.

    :param if_: Transform for the "eval" stage.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'eval', else_)


class IfTrain(IfInStage):
    """Perform *if_* if called in "train" stage, else perform *else_*.

    :param if_: Transform for the "train" stage.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'train', else_)
