from lf import trans
from lf.transform import AtomicTransform
from lf.dumptools import inspector

from lf.wrap import _wrap_class


def _maketrans(attr):
    class T(AtomicTransform):
        """Dynamically generated transform for the "this" object.

        :param args: Arguments to pass to the input's method.
        :param kwargs: Keyword arguments to pass to the input's method.
        """

        def __init__(self, *args, **kwargs):
            call_info = inspector.CallInfo()
            self.__args = args
            self.__kwargs = kwargs
            caller_frameinfo = inspector.outer_caller_frameinfo(__name__)
            call_info = inspector.CallInfo(caller_frameinfo)
            call = inspector.get_call_segment_from_frame(caller_frameinfo.frame)
            AtomicTransform.__init__(
                self,
                call=call,
                call_info=call_info,
            )

        def __call__(self, input_):
            return getattr(input_, attr)(*self.__args, **self.__kwargs)
    return T


class _This:
    """Transform factory for capturing attributes/ get-items. """

    def __getitem__(self, item):
        return _maketrans('__getitem__')(item)

    def __getattr__(self, attr):
        return _maketrans(attr)


this = _This()
