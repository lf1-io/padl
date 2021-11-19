"""Extra useful Transforms. """

from collections import OrderedDict
from typing import Optional

from padl.transforms import ClassTransform, Identity, Transform, Mode, builtin_identity


class IfInMode(ClassTransform):
    """Perform *if_* if called in mode *target_mode*, else perform *else_*.

    Example:

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

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode == self.target_mode:
            return self.if_.pd_call_transform(args)
        return self.else_.pd_call_transform(args)

    def _pd_get_stages(self, input_components=0):
        transforms = [self.if_, self.else_]
        splits = ([], [], [])
        # we need one component info per sub-transform - if it's not a list that means
        # all are the same - we make it a list
        input_components_ = input_components
        if not isinstance(input_components_, list):
            input_components_ = [input_components for _ in range(len(transforms))]

        # go through the sub-transforms ...
        output_components = []
        has_batchify = False
        for transform_, input_component in zip(transforms, input_components_):
            sub_output_components, sub_splits, sub_has_batchify = \
                transform_._pd_get_splits(input_component)
            has_batchify = has_batchify or sub_has_batchify
            output_components.append(sub_output_components)
            for split, sub_split in zip(splits, sub_splits):
                split.append(sub_split)

        cleaned_splits = tuple(
            builtin_identity if all(isinstance(s, Identity) for s in split) else split
            for split in splits
        )

        final_splits = tuple(
            IfInMode(
                if_=s[0],
                target_mode=self.target_mode,
                else_=s[1]
            ) if isinstance(s, list) else s for s in cleaned_splits
        )

        return output_components, final_splits, has_batchify


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
