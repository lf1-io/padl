import torch

from lf.lf import Transform, Identity


class Unbatchify(Transform):
    """Remove batch dimension (inverse of Batchify).

    :param dim: batching dimension
    """

    def __init__(self, module=None, stack=None, lf_name=None, dim=0):
        super().__init__(module, stack, lf_name=lf_name)
        self.dim = dim

    def __call__(self, args):
        assert self._lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if self._lf_stage != 'infer':
            return args
        if isinstance(args, tuple):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.squeeze(self.dim)
        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(Transform):
    """Add a batch dimension at dimension *dim*. During inference, this unsqueezes
    tensors and, recursively, tuples thereof.

    :param dim: batching dimension
    """

    def __init__(self, module=None, stack=None, lf_name=None, dim=0):
        super().__init__(module, stack, lf_name=lf_name)
        self.dim = dim

    def __call__(self, args):
        assert self._lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if self._lf_stage != 'infer':
            return args
        if type(args) in {tuple, list}:
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim)
        if type(args) in [float, int]:
            return torch.tensor([args])
        raise TypeError('only tensors and tuples of tensors recursively supported...')


class IfInStage(Transform):
    """Perform *t* if self.stage is *target_stage*, else perform *else_*.

    :param cond_trans: transform for the if part
    :param target_stage: stage {'train', 'eval', 'infer'}
    :param else_: transform for the else_ part
    """

    def __init__(self, cond_trans, target_stage, else_=None):
        if else_ is None:
            else_ = Identity()
        super().__init__(
            cond_trans=cond_trans,
            target_stage=target_stage,
            else_=else_,
        )
        self._mapdevice = set.union(*[t.lf_mapdevice for t in [self.t, self.else_]])

    def _versions(self):
        versions = self.cond_trans._versions().copy()
        if self.else_ is not None:
            versions.update(self.else_._versions())
        return versions

    def _clone(self):
        return type(self)(self.cond_trans, self.target_stage, self.else_)

    def stage_transform(self, stage):
        if self.target_stage == stage:
            return self.cond_trans.stage_transform(stage)
        return self.else_.stage_transform(stage)

    def has_stage_switch(self):
        return True

    def infer_transform(self):
        return self.stage_transform('infer')

    def train_transform(self):
        return self.stage_transform('train')

    def eval_transform(self):
        return self.stage_transform('eval')

    def do(self, *args):
        if self.lf_stage == self.target_stage:
            return self.cond_trans._do(*args)
        return self.else_._do(*args)

    def repr(self):
        lines = ['If{}{}:'.format(self.target_stage[0].upper(), self.target_stage[1:])]
        lines.extend(['    ' + x for x in self.cond_trans.repr().split('\n')])
        lines.append('Else:')
        lines.extend(['    ' + x for x in self.else_.repr().split('\n')])
        return '\n'.join(lines)

    @property
    def trans(self):
        trans = type(self)(cond_trans=self.cond_trans.trans, target_stage=self.target_stage,
                           else_=self.else_.trans)
        trans.lf_stage = self.lf_stage
        return trans

    def _lf_forward_part(self):
        trans = type(self)(cond_trans=self.cond_trans.lf_forward, target_stage=self.target_stage,
                           else_=self.else_.lf_forward)
        trans.lf_stage = self.lf_stage
        return trans

    @property
    def lf_postprocess(self):
        trans = type(self)(cond_trans=self.cond_trans.lf_postprocess, target_stage=self.target_stage,
                           else_=self.else_.lf_postprocess)
        trans.lf_stage = self.lf_stage
        return trans

    @property
    def postprocess_with_fixed_stage(self) -> "Transform":
        """ Get the postprocess transform while fixing it to the current stage (removing all
        branches that belong to a different stage). """
        if self.lf_stage == self.target_stage:
            return self.cond_trans.postprocess_with_fixed_stage
        return self.else_.postprocess_with_fixed_stage

    def astr(self, with_type=True, compact=True, with_types=False):
        return f'if {self.target_stage}: ({self.cond_trans.astr(False, True)}) else: ({self.else_.astr(False, True)})'

    @property
    def end_trans(self):
        if self.target_stage == 'infer':
            return self.cond_trans
        return Identity()


def IfInfer(t, else_=None):
    """Call transform *t* if infer stage otherwise call *else_*.

    :param t: transform for infer phase
    :param else_: transform otherwise
    """
    return IfInStage(t, 'infer', else_=else_)


def IfTrain(t, else_=None):
    """Perform transform *t* if train stage otherwise call *else_*.

    :param t: transform for train phase
    :param else_: transform otherwise
    """
    return IfInStage(t, 'train', else_=else_)


def IfEval(t, else_=None):
    """Perform transform *t* if eval stage otherwise call *else_*.

    :param t: transform for eval phase
    :param else_: transform otherwise
    """
    return IfInStage(t, 'eval', else_=else_)