"""Extra useful Transforms. """

from collections import OrderedDict
from typing import Optional

from padl.transforms import ClassTransform, Identity, Transform, Stage, builtin_identity


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

    def __call__(self, args):
        assert Transform.pd_stage is not None, ('Stage is not set, use infer_apply, eval_apply '
                                                'or train_apply instead of calling the transform '
                                                'directly.')

        if Transform.pd_stage == self.target_stage:
            return self.if_.pd_call_transform(args)
        return self.else_.pd_call_transform(args)

    def _pd_get_splits(self, input_components=0):
        if self._pd_splits is None or self._pd_splits[0][0] != input_components:
            transforms = [self.if_, self.else_]
            splits = ([], [], [])
            # we need one component info per sub-transform - if it's not a list that means
            # all are the same - we make it a list
            input_components_ = input_components
            if not isinstance(input_components_, list):
                input_components_ = [input_components for _ in range(len(transforms))]

            # go through the sub-transforms ...
            output_components = []
            for transform_, input_component in zip(transforms, input_components_):
                (_, sub_output_components), subsplits = transform_._pd_get_splits(input_component)
                output_components.append(sub_output_components)
                for split, subsplit in zip(splits, subsplits):
                    split.append(subsplit)

            cleaned_splits = tuple(
                builtin_identity if all(isinstance(s, Identity) for s in split) else split
                for split in splits
            )

            final_splits = tuple(
                IfInStage(
                    if_=s[0],
                    target_stage=self.target_stage,
                    else_=s[1]
                ) if isinstance(s, list) else s for s in cleaned_splits
            )

            self._pd_splits = ((input_components, output_components), final_splits)
        return self._pd_splits


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
