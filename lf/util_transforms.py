from lf.transform import BuiltinTransform, Identity, Transform


class IfInStage(BuiltinTransform):
    """Perform *t* if self.stage is *target_stage*, else perform *else_*."""

    def __init__(self, if_, target_stage, else_=None):
        """
        :param if_: transform for the if part
        :param target_stage: stage {'train', 'eval', 'infer'}
        :param else_: transform for the else_ part
        """
        super().__init__('lf.IfInStage()')
        if else_ is None:
            else_ = Identity()
        self.if_ = if_
        self.else_ = else_
        self.target_stage = target_stage

        self._lf_component = set.union(*[t.lf_component for t in [self.if_, self.else_]])

    @property
    def lf_has_stage_switch(self):
        return True

    def __call__(self, *args):
        if Transform.lf_stage == self.target_stage:
            return self.if_(*args)
        return self.else_(*args)

    def repr(self):
        lines = ['If{}{}:'.format(self.target_stage[0].upper(), self.target_stage[1:])]
        lines.extend(['    ' + x for x in self.if_.repr().split('\n')])
        lines.append('Else:')
        lines.extend(['    ' + x for x in self.else_.repr().split('\n')])
        return '\n'.join(lines)

    @property
    def lf_preprocess(self):
        return type(self)(
            if_=self.if_.lf_preprocess,
            target_stage=self.target_stage,
            else_=self.else_.lf_preprocess
        )

    def _lf_forward_part(self):
        return type(self)(
            if_=self.if_.lf_forward,
            target_stage=self.target_stage,
            else_=self.else_.lf_forward
        )

    @property
    def lf_postprocess(self):
        return type(self)(
            if_=self.if_.lf_postprocess,
            target_stage=self.target_stage,
            else_=self.else_.lf_postprocess
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