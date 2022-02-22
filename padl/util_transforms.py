"""Extra useful Transforms. """

from textwrap import indent
from collections import OrderedDict
from typing import Optional, Union, List, Tuple

from padl.transforms import ClassTransform, Identity, Transform, Mode, identity


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

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        if_repr = f'if pd_mode == "{self.target_mode}":\n\n' + indent(
            self.if_._pd_longrepr(marker=None), '   ')
        else_repr = f'\n\nelse:\n\n' + indent(self.else_._pd_longrepr(marker=None), '   ')
        if marker:
            return marker[1] + '\n\n' + if_repr + else_repr
        return if_repr + else_repr

    def __call__(self, args):
        assert Transform.pd_mode is not None, ('Mode is not set, use infer_apply, eval_apply '
                                               'or train_apply instead of calling the transform '
                                               'directly.')

        if Transform.pd_mode == self.target_mode:
            return self.if_._pd_unpack_args_and_call(args)
        return self.else_._pd_unpack_args_and_call(args)

    def _pd_splits(self, input_components=0):
        # pylint: disable=protected-access
        if_output_components, if_splits, if_has_batchify, if_has_unbatchify = \
            self.if_._pd_splits(input_components)
        else_output_components, else_splits, else_has_batchify, else_has_unbatchify = \
            self.else_._pd_splits(input_components)

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

        return if_output_components_reduced, final_splits, \
               if_has_batchify or else_has_batchify, if_has_unbatchify or else_has_unbatchify


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
    :param finally_transform: Transform to carry on the `finally` clause of the `try` statement.
    :param pd_name: The Transform's name.
    """

    def __init__(self,
                 transform: Transform,
                 catch_transform: Transform,
                 exceptions: Union[List, Tuple, Exception],
                 else_transform: Transform = identity,
                 finally_transform: Transform = identity,
                 pd_name: str = None):

        if not isinstance(exceptions, (tuple, list)):
            exceptions = (exceptions,)
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

    def _pd_splits(self, input_components=0):
        try_output_components, _, _, _ = self.transform._pd_splits(input_components)
        catch_output_components, _, _, _ = self.catch_transform._pd_splits(input_components)
        else_output_components, _, _, _ = self.else_transform._pd_splits(input_components)

        input_components_reduced = self._pd_merge_components(input_components)
        try_output_components_reduced = self._pd_merge_components(try_output_components)
        catch_output_components_reduced = self._pd_merge_components(catch_output_components)
        else_output_components_reduced = self._pd_merge_components(else_output_components)
        components = [try_output_components_reduced, catch_output_components_reduced,
                      else_output_components_reduced]
        assert all(isinstance(component, int) for component in components) \
               and len(set(components)) == 1, \
            'Try Transform cannot contain transforms that have multiple stages.'

        final_splits = tuple(
            self if i == components[0]
            else identity
            for i in range(3)
        )

        return input_components_reduced, final_splits, False, False

    def _repr_exceptions(self):
        return '(' + ', '.join([exc.__name__ for exc in self.exceptions]) + ')'

    def _pd_longrepr(self, formatting=True, marker=None) -> str:
        tr_repr = f'try:\n\n' + indent(self.transform._pd_longrepr(marker=None), '   ')
        catch_tr_repr = f'\n\nexcept ' + self._repr_exceptions() + ':\n\n' + indent(
            self.catch_transform._pd_longrepr(marker=None), '   ')
        if isinstance(self.else_transform, Identity):
            else_tr_repr = ''
        else:
            else_tr_repr = f'\n\nelse:\n\n' + indent(
                self.else_transform._pd_longrepr(marker=None), '   ')
        if isinstance(self.finally_transform, Identity):
            finally_repr = ''
        else:
            finally_repr = f'\n\nfinally:\n\n' + indent(
                self.finally_transform._pd_longrepr(marker=None), '   ')
        if marker:
            return marker[1] + '\n\n' + tr_repr + catch_tr_repr + else_tr_repr + finally_repr
        return tr_repr + catch_tr_repr + else_tr_repr + finally_repr

    def __call__(self, args):
        try:
            output = self.transform._pd_unpack_args_and_call(args)
        except self.exceptions:
            output = self.catch_transform._pd_unpack_args_and_call(args)
        else:
            output = self.else_transform._pd_unpack_args_and_call(output)
        finally:
            self.finally_transform._pd_unpack_args_and_call(args)
        return output
