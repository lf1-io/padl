""" Utility functions used in the "lf.interactive" package. """
from functools import wraps
from inspect import stack
from itertools import zip_longest

from lf import transforms as lf


def plen(value):
    """
    :param value:
    :return:
    """
    return int(len(value) - value.count('\x1b') * 4.5 + value.count('\x1b[1'))


def stringblock(string_value, padding=0):
    """
    Fill all lines of *string_value* with whitespace such that all lines have the same length.
    """
    lines = string_value.split('\n')

    length = max(plen(line) for line in lines) + padding

    return '\n'.join([line + ' ' * (length - plen(line)) for line in lines])


def arrow(input_string: str, where: str) -> str:
    """ Add an arrow on the right of all lines in string *s* that start with *where*. """
    lines = input_string.split('\n')
    out_lines = []
    for line in lines:
        if line.lstrip().startswith(where):
            r_strip = line.rstrip()
            o_line = r_strip + blue(' ←' + '⎼' * (len(line) - len(r_strip) - 2))
            out_lines.append(o_line)
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)


def crop(input_string, maxwidth):
    """
    crop *input_string*

    :param input_string: input string
    :param maxwidth: max width
    """
    res = []
    lines = input_string.split('\n')
    for _, line in enumerate(lines):
        if plen(line) > maxwidth:
            res.append(line[:maxwidth - 4] + ' ...')
        else:
            res.append(line)
    return '\n'.join(res)


def larrow(input_string: str, where: str, length=4) -> tuple:
    """ Add an arrow on the right of all lines in string *input_string* that start with *where*. """
    lines = input_string.split('\n')
    out_lines = []
    line_num = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(where):
            r_strip = line.rstrip()
            o_line = blue('⎼' * (length - 2) + '→ ') + r_strip
            out_lines.append(o_line)
            line_num = i
        else:
            out_lines.append(' ' * length + line)
    return '\n'.join(out_lines), line_num


def blue(input_string: str):
    """ Make *input_string* blue. """
    return '\u001b[34m' + input_string + '\u001b[0m'


def bracket(input_string: str, padding: int = 2):
    """ Add a big bracket to the left of *input_string*. """
    lines = input_string.split('\n')
    res = blue('⎡') + ' ' * padding + lines[0] + '\n'
    res += '\n'.join(blue('⎢') + ' ' * padding + line for line in lines[1:-1])
    res += '\n' + blue('⎣') + ' ' * padding + lines[-1]
    return res


def rbracket(input_string: str, connect_line: int, padding: int = 2):
    """ Add a big bracket to the right of *input_string*. """
    lines = input_string.split('\n')
    if len(lines) < connect_line + 1:
        lines += [' ' * (len(lines[-1]))] * (connect_line + 2 - len(lines))
    res = lines[0] + ' ' * padding + blue('⎫') + '\n'
    res += '\n'.join(line + ' ' * padding + blue('⎬' if i + 1 == connect_line else '⎪')
                     for i, line in enumerate(lines[1:-1]))
    res += '\n' + lines[-1] + ' ' * padding + blue('⎭')
    return res


def connect(string_a, string_b, where):
    """ Connect *string_b* and *string_a* at the line that starts with *where*. """
    string_b = stringblock(string_b, 3)
    string_a, line_num = larrow(string_a, where)
    string_b = rbracket(string_b, line_num)
    las = string_b.split('\n')
    lbs = string_a.split('\n')
    lbs += ['\n'] * (len(las) - len(lbs) - 2)
    string_a = '\n'.join(lbs)
    lbs = string_a.split('\n')

    res = []
    for la_, lb_ in zip_longest(las, lbs):
        if la_ is None:
            la_ = ' ' * plen(las[0])
        if lb_ is None:
            lb_ = ''
        res.append(la_ + lb_)
    return '\n'.join(res)


def lconnect(string_a, string_b, where):
    """ Connect *string_a* and *string_b* at the line that starts with *where*. """
    string_a = stringblock(string_a, 10)
    string_a = arrow(string_a, where)
    las = string_a.split('\n')
    lbs = string_b.split('\n')
    lbs += ['\n'] * (len(las) - len(lbs) - 2)
    string_b = '\n'.join(lbs)
    string_b = bracket(string_b)
    lbs = string_b.split('\n')

    res = []
    for la_, lb_ in zip_longest(las, lbs):
        if la_ is None:
            la_ = ' ' * plen(las[0])
        if lb_ is None:
            lb_ = ''
        res.append(la_ + lb_)
    return '\n' + '\n'.join(res)


def connect_chain(chain):
    """
    :param chain: chain
    :return:
    """
    if len(chain) == 1:
        return str(chain[0][0])
    return connect(str(chain[0][0]), connect_chain(chain[1:]), chain[1][1])


class Tracer:
    """Tracer"""
    def __init__(self):
        self.start = None
        self.current = None
        self.old_do = None

    def append(self, trans):
        entry = {'parent': self.current, 'trans': trans, 'children': []}
        self.current['children'].append(entry)
        return entry

    def new_layer(self):
        self.current = self.current['children'][-1]

    def up(self):
        self.current = self.current['parent']

    def decide(self, trans, parent):
        if self.start is None:
            self.start = {'trans': trans, 'children': [], 'parent': 'root'}
            self.current = self.start
            return self.start
        if parent is self.current['trans']:
            return self.append(trans)
        if self.current['children'] and parent is self.current['children'][-1]['trans']:
            self.new_layer()
            return self.append(trans)
        self.up()
        return self.append(trans)

    def __enter__(self):
        self.old_do = lf.Transform._do

        @wraps(self.old_do)
        def new_do(self_ref, args, *args_, **kwargs):
            """
            Do for tracer
            :param self_ref: ref to self
            :param args: args for transforms
            :param args_:
            :param kwargs:
            :return:
            """
            self_ref._trace_args = args
            parent = None
            for layer in stack():
                try:
                    potential_parent = layer.frame.f_locals['self']
                    if isinstance(potential_parent, lf.Transform) and \
                            potential_parent is not self_ref:
                        parent = layer.frame.f_locals['self']
                        break
                except KeyError:
                    continue
            entry = self.decide(self_ref, parent)
            out = self.old_do(self_ref, args, *args_, **kwargs)
            entry['in'] = args
            entry['out'] = out
            return out

        lf.Transform._do = new_do
        return self

    def __exit__(self, *args, **kwargs):
        lf.Transform._do = self.old_do
