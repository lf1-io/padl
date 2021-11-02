"""Utilities for printing. """

from reprlib import repr as shortrepr
from typing import Any, List

from collections import UserString


class FStr(UserString):
    use = True

    def __init__(self, lines, bolds=None, greens=None):
        if isinstance(lines, str):
            lines = lines.split('\n')
        if bolds is None:
            bolds = [[] for _ in lines]
        if greens is None:
            greens = [[] for _ in lines]
        super().__init__('\n'.join(lines))
        self.lines = lines
        self.bolds = bolds
        self.greens = greens

    def __add__(self, other):
        # body = '\n'.join(self.lines) + '\n'.join(other.lines)
        ...

    @classmethod
    def span(cls, *inputs):
        words = []
        for input in inputs:
            bolds = ()
            greens = ()
            if inputs[:4] == 'BO!' or inputs[:4] == 'BG!':
                bolds = range(0, len(input))
            if input[:4] == 'GO!' or inputs[:4] == 'BG!':
                greens = range(0, len(input))
            words.append(
                FStr([input], bolds=bolds, greens=greens)
            )
        return sum(words)

    def repr(self):
        output = ''
        for i, line in enumerate(self.lines):
            for j, c in enumerate(list(line)):
                if j in self.bolds[i]:
                    c = make_bold(c)
                if j in self.greens[i]:
                    c = make_green(c)
                output += c
            output += '\n'
        return output

    def __repr__(self):
        if not self.use:
            return super().__repr__()
        return self.repr()


def combine_multi_line_strings(strings: List[str]):
    """Combine multi line strings, where every character takes precedence over space.
    (Image space is transparent and superimposing strings on top of one another.)

    :param strings: list of multi-line strings
    """
    strings = [x.split('\n') for x in strings]
    length = max([len(x) for x in strings])
    lines = []
    for i in range(length):
        parts = [x[i] for x in strings if len(x) >= i + 1]
        line = list(' ' * max([len(x) for x in parts]))
        for j in range(len(line)):
            for part in parts:
                if len(part) >= j + 1:
                    if part[j] != ' ':
                        line[j] = part[j]
        lines.append(''.join(line))
    return '\n'.join(lines)


def create_reverse_arrow(start_left: int, finish_right: int, n_initial_rows: int,
                         n_final_rows: int):
    """Create an ascii arrow.

    :param start_left: X-coordinate where the last line arrow ends.
    :param finish_right: X-coor where the first line arrow starts.
    :param n_initial_rows: Y-padding.
    :param n_final_rows: Y-padding.

    Example:

        ____________|
        |
        v
    """
    initial = ''
    for _ in range(n_final_rows - 1):
        initial += ' ' * (start_left + finish_right) + '│' + '\n'

    final = ''
    for _ in range(n_initial_rows - 1):
        final += ' ' * start_left + '│' + '\n'

    underscore_line = list('_' * finish_right)
    if underscore_line:
        underscore_line[0] = ' '
    underscore_line = ''.join(underscore_line)

    output = initial \
        + ' ' * start_left + underscore_line + '│' + ' ' + '\n' \
        + final \
        + ' ' * start_left + '▼'
    output = '\n'.join(output.split('\n')[1:])

    return output


def make_bold(x):
    """Make input bold

    :param x: string input
    """
    return f'\33[1m{x}\33[0m'


def make_green(x):
    """Make input green

    :param x: string input
    """
    return f'\33[32m{x}\33[0m'


def create_arrow(start_left: int, finish_right: int, n_initial_rows: int, n_final_rows: int) -> str:
    """
    Create an ascii arrow.

    :param start_left: X-coordinate where the last line arrow ends.
    :param finish_right: X-coor where the first line arrow starts.
    :param n_initial_rows: Y-padding.
    :param n_final_rows: Y-padding.

    Example:

        ____________|
        |
        v
    """
    initial = ''
    for _ in range(n_initial_rows - 1):
        initial += ' ' * start_left + '│' + '\n'
    final = ''
    for _ in range(n_final_rows - 1):
        final += ' ' * (start_left + finish_right) + '│' + '\n'
    bend_start = '└' if finish_right else '│'
    bend_end = '┐' if finish_right else ''
    return initial + ' ' * start_left + bend_start \
        + '─' * (finish_right - 1) + bend_end + '\n' \
        + final \
        + ' ' * (start_left + finish_right) + '▼'


def format_argument(value: Any) -> str:
    """Format an argument value for printing."""
    return shortrepr(value)


def visible_len(value: str) -> int:
    """The number of visible characters in a string.

    Use this to get the length of strings that contain escape sequences.
    """
    return int(len(value) - value.count('\x1b') * 4.5 + value.count('\x1b[1'))


def create_box(w, h, padding_left=0):
    """
    See https://en.wikipedia.org/wiki/Box-drawing_character
    """
    lines = [
        '┌' + ('─' * (w - 2)) + '┐',
        *[('│' + (' ' * (w - 2)) + '│') for _ in range(h - 2)],
        ('└' + ('─' * (w - 2)) + '┘')
    ]
    for i, x in enumerate(lines):
        lines[i] = ' ' * padding_left + x
    return '\n'.join(lines)


def put_box_around_string(string):
    string = string.split('\n')
    max_length = max([visible_len(x) for x in string])
    string = [' ' * 2] + [' ' * 2 + x for x in string]
    box = create_box(max_length + 4, len(string) + 1)
    string = '\n'.join(string)
    import pdb; pdb.set_trace()
    return combine_multi_line_strings([box, string])


def pad_string_left(string, n):
    return '\n'.join([' ' * n + x for x in string.split('\n')])