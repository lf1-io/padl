"""Extra useful Transforms. """

from collections import OrderedDict
from typing import Optional, Union, List, Tuple

from padl.transforms import ClassTransform, Identity, Transform, Mode


class IfInMode(ClassTransform):
    """Perform *if_* if called in mode *target_mode*, else perform *else_*.

    Example:

        >>> from padl import transform
        >>> a = transform(lambda x: x + 10)
        >>> b = transform(lambda x: x * 10)
        >>> iis = IfInMode(a, 'infer', b)
        >>> iis.infer_apply(1)
        11
        >>> list(iis.eval_apply([1]))
        [100]

    :param if_: Transform to apply when the mode matches.
    :param target_mode: Mode (one of 'train', 'eval', 'infer').
    :param else_: Transform to apply when the mode doesn't match (defaults to identity transform).
    """

    pd_dont_dump_code = True

    def __init__(self, if_: Transform, target_mode: Mode, else_: Optional[Transform] = None):
        super().__init__(arguments=OrderedDict([('if_', if_),
                                                ('target_mode', target_mode),
                                                ('else_', else_)]))

        assert target_mode in ('train', 'eval', 'infer'), "Target mode can only be train, " \
                                                          "eval or infer"

        if else_ is None:
            else_ = Identity()

        self.if_ = if_
        self.else_ = else_
        self.target_mode = target_mode

        self._pd_stage = set.union(*[t.pd_stage for t in [self.if_, self.else_]])

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode == self.target_mode:
            return self.if_.pd_call_transform(args)
        return self.else_.pd_call_transform(args)

    def _pd_preprocess_part(self):
        pre = IfInMode(
            if_=self.if_.pd_preprocess,
            target_mode=self.target_mode,
            else_=self.else_.pd_preprocess,
        )
        return pre

    def _pd_forward_part(self):
        forward = IfInMode(
            if_=self.if_.pd_forward,
            target_mode=self.target_mode,
            else_=self.else_.pd_forward,
        )
        return forward

    def _pd_postprocess_part(self):
        post = IfInMode(
            if_=self.if_.pd_postprocess,
            target_mode=self.target_mode,
            else_=self.else_.pd_postprocess,
        )
        return post


class IfInfer(IfInMode):
    """Perform *if_* if called in "infer" mode, else perform *else_*.

    :param if_: Transform for the "infer" mode.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'infer', else_)


class IfEval(IfInMode):
    """Perform *if_* if called in "eval" mode, else perform *else_*.

    :param if_: Transform for the "eval" mode.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'eval', else_)


class IfTrain(IfInMode):
    """Perform *if_* if called in "train" mode, else perform *else_*.

    :param if_: Transform for the "train" mode.
    :param else_: Transform otherwise (defaults to the identity transform).
    """

    def __init__(self, if_: Transform, else_: Optional[Transform] = None):
        super().__init__(if_, 'train', else_)


class Try(ClassTransform):
    """ Perform *transform*. If this fails with any exception from *exceptions*, perform
    *catch_transform*. If *transform* is completed successfully, *else_transform* is performed
    with the output of *transform*. Regardless of any error occurring on the other Transform
    (or not), *finally_transform* is carried out. No change of mode can happen inside any of these
    Transform.

    :param transform: Transform to try.
    :param catch_transform: Transform to fall back on.
    :param exceptions: Catch conditions.
    :param else_transform: Transform to carry on the `else` clause of the `try` statement.
    :param finally_transform: Transform to carry on on the `finally` clause of the `try` statement.
    :param pd_name: The Transform's name.
    """

    def __init__(self,
                 transform: Transform,
                 catch_transform: Transform,
                 exceptions: Union[List, Tuple, Exception],
                 else_transform: Transform = Identity(),
                 finally_transform: Transform = Identity(),
                 pd_name: str = None):

        if not isinstance(exceptions, (tuple, list)):
            exceptions = (exceptions, )
        exceptions = tuple(exceptions)
        for exception in exceptions:
            assert issubclass(exception, Exception)
        super().__init__(pd_name=pd_name,
                         arguments=OrderedDict([('transform', transform),
                                                ('catch_transform', catch_transform),
                                                ('exceptions', exceptions),
                                                ('else_transform', else_transform),
                                                ('finally_transform', finally_transform)]))
        self.transform = transform
        self.catch_transform = catch_transform
        self.exceptions = exceptions
        self.else_transform = else_transform
        self.finally_transform = finally_transform
        self._pd_stage = set.union(self.transform.pd_stage,
                                   self.catch_transform.pd_stage,
                                   self.else_transform.pd_stage,
                                   self.finally_transform.pd_stage)

        assert len(self._pd_stage) == 1, 'Try Transform cannot contain transforms that have ' \
                                            'multiple components.'

    @property
    def pd_stage(self):
        return self._pd_stage

    def __call__(self, args):
        try:
            output = self.transform.pd_call_transform(args)
        except self.exceptions:
            output = self.catch_transform.pd_call_transform(args)
        else:
            output = self.else_transform.pd_call_transform(output)
        finally:
            self.finally_transform.pd_call_transform(args)
        return output
