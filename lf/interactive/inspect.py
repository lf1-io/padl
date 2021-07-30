""" Experimental. """
from functools import wraps
from inspect import stack

from warnings import warn
from lf import transforms as lf

from . import utils


warn('These tools are experimental, use at your own risk.')


def names():
    """
    Name transforms using the names of the variables they were assigned to.

    Example:

    >>> a = Transform()
    >>> b = Transform()
    >>> c = a >> b
    >>> names()
    >>> a._name
    'a'
    >>> b._name
    'b'
    >>> c._name
    'c'
    """
    caller_frame = stack()[1]
    for k, value in caller_frame.frame.f_globals.items():
        if k.startswith('_'):
            continue
        if isinstance(value, lf.Transform) and value._name is None:
            value._name = k


def autonames():
    """
    Automatically name transforms using the name of the variable they're assigned to.
    """

    @property
    def name(self):
        """Assign name to transforms"""
        if self._name is not None:
            return self._name

        for caller_frame in stack()[::-1]:
            for k, value in caller_frame.frame.f_globals.items():
                if self is value and not k.startswith('_'):
                    self._name = k
                    break
            if self._name is not None:
                break
        return self._name

    @name.setter
    def name(self, x):
        """
        Assign name to transform

        :param x: name
        """
        self._name = x

    lf.Transform.name = name


def get_response(x, trans, chain, tracer_stack=None, maxwidth=None):
    """
    Get next response for interactive transform inspector
    :param x: user input
    :param trans: transform
    :param chain: chain of transform
    :param tracer_stack:
    :param maxwidth:
    """
    if tracer_stack is None:
        res = inspect(chain + [(trans[int(x)], f'{x}:')], maxwidth=maxwidth)
    else:
        res = inspect(chain + [(trans[int(x)], f'{x}:')],
                      tracer_stack['children'][int(x)], maxwidth)
    return res


def inspect(chain, tracer_stack=None, maxwidth=100):
    """ Start an interactive transform inspector.

    :param chain: chain
    :param tracer_stack: tracer stack
    :param maxwidth: max width
    """
    if isinstance(chain, lf.Transform):
        return inspect([(chain, None)], tracer_stack=tracer_stack, maxwidth=maxwidth)
    trans = chain[-1][0]
    msg = utils.connect_chain(chain)
    if tracer_stack is not None:
        msg = '\n\n' + '    ' + \
            str(tracer_stack['in']) + '\n    ↓\n\n' + msg + \
            '\n\n    ↓\n' + '    ' + str(tracer_stack['out'])

    return_dict = {'u': 'u',
                   'q': 'q',
                   'r': trans,
                   }
    while True:
        print(utils.crop(msg, maxwidth))
        print()
        x = input('> ')
        if len(x) == 0:
            continue
        if x[0] in return_dict:
            return return_dict[x[0]]
        if x.startswith('n'):
            trans.name = x.split(' ', 1)[1]
        else:
            try:
                res = get_response(x, trans, chain, tracer_stack, maxwidth)
                if isinstance(res, lf.Transform):
                    return res
                elif res != 'u':
                    break
            except (ValueError, KeyError, IndexError):
                print()
                print('Command not understood.\n'
                      '   u: go up\n'
                      '   r: return this\n'
                      '   some integer: go down\n'
                      '   q: quit')
        msg = utils.connect_chain(chain)
        if tracer_stack is not None:
            msg = '\n\n' + '    ' + \
                str(tracer_stack['in']) + '\n    ↓\n\n' + msg + \
                '\n\n    ↓\n' + '    ' + str(tracer_stack['out'])


def trace(trans, input, maxwidth=100):
    """
    Trace

    :param trans: transform
    :param input: input to the transform
    :param maxwidth: max width
    """
    with utils.Tracer() as a_tracer:
        trans(input)
        return inspect(trans, a_tracer.start, maxwidth)


class _Debug:
    """Debug"""
    def __init__(self):
        self.trans = None
        self.args = None

    def repeat(self) -> None:
        self.trans.do(self.args)

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
        pos = len(lf.core._lf_trace) - 1
        msg = lf.core._lf_trace[pos][0]
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
                pos, msg = self._down_step(pos, lf.core._lf_trace)
            elif x == 'u':
                pos, msg = self._up_step(pos, lf.core._lf_trace)
            elif x == 'q':
                self.args = lf.core._lf_trace[pos][2]
                self.trans = lf.core._lf_trace[pos][3]
                break
            elif x == 'w':
                msg = lf.core._lf_trace[pos][1]
            elif x == 'i':
                msg = lf.core._lf_trace[pos][2]
            elif x == 'r':
                self.args = lf.core._lf_trace[pos][2]
                self.trans = lf.core._lf_trace[pos][3]
                self.repeat()

    def _down_step(self, pos, lf_trace):
        """
        :param pos: position
        :param lf_trace:
        :return:
        """
        if pos > 0:
            pos -= 1
            return pos, lf_trace[pos][0]
        return pos, 'Reached the bottom.'

    def _up_step(self, pos, lf_trace):
        """
        :param pos: position
        :param lf_trace: lf_trace
        :return:
        """
        if pos < len(lf_trace) - 1:
            pos += 1
            return pos, lf_trace[pos][0]
        return pos, 'Reached top level.'

debug = _Debug()
