"""Extra useful Transforms. """

from collections import OrderedDict
from typing import Optional

from padl.transforms import ClassTransform, Identity, Transform, Mode


class IfInMode(ClassTransform):
    """Perform *if_* if called in mode *target_mode*, else perform *else_*.

    Example:

        >>> from padl import transform
        >>> a = transform(lambda x: x + 10)
        >>> b = transform(lambda x: x * 99)
        >>> iim = IfInMode(a, 'infer', b)
        >>> iim.infer_apply(1)
        11
        >>> list(iim.eval_apply([1]))
        [99]

    :param if_: Transform to apply when the mode matches.
    :param target_mode: Mode (one of 'train', 'eval', 'infer').
    :param else_: Transform to apply when the mode doesn't match (defaults to identity transform).
    """

    def __init__(self, if_: Transform, target_mode: Mode, else_: Optional[Transform] = None):
        assert target_mode in ('train', 'eval', 'infer'), "Target mode can only be train, " \
                                                          "eval or infer"
        super().__init__(arguments=OrderedDict([('if_', if_),
                                                ('target_mode', target_mode),
                                                ('else_', else_)]))

        if else_ is None:
            else_ = Identity()

        self.if_ = if_
        self.else_ = else_
        self.target_mode = target_mode

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode == self.target_mode:
            return self.if_.pd_call_transform(args)
        return self.else_.pd_call_transform(args)

    def _pd_get_splits(self, input_components=0):
        # pylint: disable=protected-access
        if_output_components, if_splits, if_has_batchify = \
            self.if_._pd_get_splits(input_components)
        else_output_components, else_splits, else_has_batchify = \
            self.else_._pd_get_splits(input_components)

        if_output_components_reduced = self._pd_merge_components(if_output_components)
        else_output_components_reduced = self._pd_merge_components(else_output_components)
        assert if_output_components_reduced == else_output_components_reduced, \
            'if and else transforms have incompatible output shapes'

        final_splits = tuple(
            IfInMode(
                if_=if_split,
                target_mode=self.target_mode,
                else_=else_split
            ) if if_split != else_split
            else if_split
            for if_split, else_split in zip(if_splits, else_splits)
        )

        return if_output_components_reduced, final_splits, if_has_batchify or else_has_batchify


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
