"""Utilities. """

from padl.dumptools import inspector
from padl.print_utils import format_argument
from padl.transforms import AtomicTransform
from padl.transforms import _pd_trace


def _maketrans(attr, getitem=False):
    class T(AtomicTransform):
        """Dynamically generated transform for the "same" object.

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

        def _pd_longrepr(self, formatting=True, marker=None):
            out = self._pd_shortrepr()
            if marker:
                return out + marker[1]
            return out

        def _pd_shortrepr(self, formatting=True):
            return f'{attr}({self._formatted_args()})'

        def _pd_tinyrepr(self, formatting=True):
            return self.pd_name or attr

    return T


class _Same:
    """Transform factory for capturing attributes/ get-items. """

    def __getitem__(self, item):
        return _maketrans('__getitem__', True)(item)

    def __getattr__(self, attr):
        return _maketrans(attr)


#: Transform factory for capturing attributes/ get-items.
same = _Same()


class _Debug:
    """Debugger for padl Transforms.
    """
    def __init__(self):
        self.trans = None
        self.args = None
        self.default_msg = (
            'Defined commands are: \n'
            '   u(p): step up\n'
            '   d(own): step down\n'
            '   w(here am I?): show code position\n'
            '   i(nput): show input here\n'
            '   r(epeat): repeat here (will produce the same exception)\n'
            '   t(ransform): displays the current transform\n'
            '   h(elp): print help about the commands\n'
            '   q(uit): quit'
            '     -> this will store the input at this level in debug.args\n'
            '        and the transform in debug.trans\n'
        )

    def __call__(self) -> None:
        """
        Call me in case of error.
        User can give following input and expect response
            u(p): step up\n'
            d(own): step down\n'
            w(here am I?): show code position\n'
            i(nput): show input here\n'
            r(epeat): repeat here (will produce the same exception)\n'
            h(elp): print help about the commands\n
            q(uit): quit'
        """
        pos = len(_pd_trace) - 1
        print(self.default_msg + '\n' + _pd_trace[pos].transform_str)

        while True:
            try:
                x = input('> ')
            except IndexError:
                continue
            if x == 'd':
                pos, msg = self._down_step(pos, _pd_trace)
            elif x == 'u':
                pos, msg = self._up_step(pos, _pd_trace)
            elif x == 'q':
                self.args = _pd_trace[pos].args
                self.trans = _pd_trace[pos].transform
                break
            elif x == 'w':
                msg = _pd_trace[pos].code_position
            elif x == 'i':
                msg = _pd_trace[pos].args
            elif x == 'r':
                self.args = _pd_trace[pos].args
                self.trans = _pd_trace[pos].transform
                # This 0 is because the last element carries a problem: when adding the last
                # element to _pd_trace *Transform.pd_mode* has been already set up to None again.
                self.repeat(_pd_trace[0].pd_mode)
            elif x == 'h' or x == 'help':
                msg = self.default_msg
            elif x == 't':
                msg = _pd_trace[pos].transform_str
            else:
                i = _pd_trace[pos].args
                try:
                    code = compile(x, '', 'single')
                    exec(code)
                except Exception as err:
                    print(err)
                finally:
                    continue
            if x in {'d', 'u', 'w', 'i', 'h', 'help', 't'}:
                print(f'\n{msg}\n')

    def repeat(self, mode: str) -> None:
        assert mode in ('train', 'eval', 'infer')
        _pd_trace.clear()
        if mode == 'train':
            breakpoint()
            list(self.trans.train_apply([self.args], batch_size=len(self.args), num_workers=0))
        if mode == 'eval':
            list(self.trans.eval_apply([self.args], batch_size=len(self.args), num_workers=0))
        self.trans.infer_apply(self.args)

    @staticmethod
    def _down_step(pos, pd_trace):
        if pos > 0:
            pos -= 1
            return pos, pd_trace[pos].transform_str
        return pos, 'Reached the bottom.'

    @staticmethod
    def _up_step(pos, pd_trace):
        if pos < len(pd_trace) - 1:
            pos += 1
            return pos, pd_trace[pos].transform_str
        return pos, 'Reached top level.'


pd_debug = _Debug()
