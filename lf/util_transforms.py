from collections import OrderedDict

from lf.transforms import ClassTransform, Identity, Transform


class IfInStage(ClassTransform):
    """Perform *if_* if self.stage is *target_stage*, else perform *else_*.

    :param if_: transform for the if part
    :param target_stage: stage {'train', 'eval', 'infer'}
    :param else_: transform for the else_ part
    """

    def __init__(self, if_, target_stage, else_=None):
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

        self._td_component = set.union(*[t.td_component for t in [self.if_, self.else_]])

    def __call__(self, *args):
        assert Transform.td_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if Transform.td_stage == self.target_stage:
            return self.if_(*args)
        return self.else_(*args)

    @property
    def td_preprocess(self):
        return type(self)(
            if_=self.if_.td_preprocess,
            target_stage=self.target_stage,
            else_=self.else_.td_preprocess
        )

    def _td_forward_part(self):
        return type(self)(
            if_=self.if_.td_forward,
            target_stage=self.target_stage,
            else_=self.else_.td_forward
        )

    @property
    def td_postprocess(self):
        return type(self)(
            if_=self.if_.td_postprocess,
            target_stage=self.target_stage,
            else_=self.else_.td_postprocess
        )


def IfInfer(if_, else_=None):
    """
    :param if_: transform for infer phase
    :param else_: transform otherwise
    """
    return IfInStage(if_, 'infer', else_=else_)


def IfTrain(if_, else_=None):
    """
    :param if_: transform for train phase
    :param else_: transform otherwise
    """
    return IfInStage(if_, 'train', else_=else_)


def IfEval(if_, else_=None):
    """
    :param if_: transform for eval phase
    :param else_: transform otherwise
    """
    return IfInStage(if_, 'eval', else_=else_)
