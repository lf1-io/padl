"""Utilities. """

from padl.dumptools import inspector
from padl.print_utils import format_argument
from padl.transforms import AtomicTransform


def _maketrans(attr, getitem=False):
    class T(AtomicTransform):
        """Dynamically generated transform for the "this" object.

        :param args: Arguments to pass to the input's method.
        :param kwargs: Keyword arguments to pass to the input's method.
        """

        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
            call_info = inspector.CallInfo(caller_frameinfo)
            if getitem:
                call = inspector.get_segment_from_frame(caller_frameinfo.frame, 'getitem')
            else:
                call = inspector.get_segment_from_frame(caller_frameinfo.frame, 'call')
            AtomicTransform.__init__(
                self,
                call=call,
                call_info=call_info,
            )

        def __call__(self, args):
            return getattr(args, attr)(*self._args, **self._kwargs)

        def _formatted_args(self) -> str:
            """Format the object's init arguments for printing. """
            args_list = [format_argument(val) for val in self._args]
            args_list += [f'{key}={format_argument(val)}' for key, val in self._kwargs.items()]
            return ', '.join(args_list)

        def _pd_longrepr(self):
            return self._pd_shortrepr()

        def _pd_shortrepr(self):
            return f'{attr}({self._formatted_args()})'

        def _pd_tinyrepr(self):
            return self.pd_name or attr

    return T


class _This:
    """Transform factory for capturing attributes/ get-items. """

    def __getitem__(self, item):
        return _maketrans('__getitem__', True)(item)

    def __getattr__(self, attr):
        return _maketrans(attr)


this = _This()
