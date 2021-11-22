from padl.transforms import _pd_trace


class _Debug:
    """Debugger for padl Transforms. """
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
        print(_pd_trace[pos][0])

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
            elif x == 'h':
                msg = self.default_msg
            elif x == 't':
               msg = _pd_trace[pos][0]
            else:
                i = _pd_trace[pos][2]
                try:
                    code = compile(x, '', 'single')
                    exec(code)
                except Exception as err:
                    print(err)
                finally:
                    continue
            if x in {'d', 'u', 'w', 'i', 'h', 't'}:
                print(f'\n{msg}\n')

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
