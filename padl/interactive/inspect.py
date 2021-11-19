from padl.transforms import _pd_trace


class _Debug:
    """Debugger for padl Transforms. """
    def __init__(self):
        self.trans = None
        self.args = None

    def __call__(self) -> None:
        """
        Call me in case of error.
        User can give following input and expect response
            u(p): step up\n'
            d(own): step down\n'
            w(here am I?): show code position\n'
            i(nput): show input here\n'
            r(epeat): repeat here (will produce the same exception)\n'
            q(uit): quit'
        """
        pos = len(_pd_trace)
        breakpoint()
        msg = _pd_trace[pos][0]
        default_msg = (
                        'Command not understood.\n\n'
                        'Options are: \n'
                        '   u(p): step up\n'
                        '   d(own): step down\n'
                        '   w(here am I?): show code position\n'
                        '   i(nput): show input here\n'
                        '   r(epeat): repeat here (will produce the same exception)\n'
                        '   q(uit): quit'
                        '     -> this will store the input at this level in debug.args\n'
                        '        and the transform in debug.trans\n'
                    )
        while True:
            print()
            print(msg)
            print()
            msg = default_msg
            try:
                x = input('> ')[0]
            except IndexError:
                continue
            if x == 'd':
                pos, msg = self._down_step(pos, _pd_trace)
            elif x == 'u':
                pos, msg = self._up_step(pos, _pd_trace)
            elif x == 'q':
                self.args = _pd_trace[pos][2]
                self.trans = _pd_trace[pos][3]
                break
            elif x == 'w':
                msg = _pd_trace[pos][1]
            elif x == 'i':
                msg = _pd_trace[pos][2]
            elif x == 'r':
                self.args = _pd_trace[pos][2]
                self.trans = _pd_trace[pos][3]
                self.repeat()

    def repeat(self) -> None:
        self.trans(self.args)

    @staticmethod
    def _down_step(pos, pd_trace):
        if pos > 0:
            pos -= 1
            return pos, pd_trace[pos][0]
        return pos, 'Reached the bottom.'

    @staticmethod
    def _up_step(pos, pd_trace):
        if pos < len(pd_trace) - 1:
            pos += 1
            return pos, pd_trace[pos][0]
        return pos, 'Reached top level.'


pd_debug = _Debug()
