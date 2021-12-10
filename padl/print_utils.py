"""Utilities for printing. """

from reprlib import repr as shortrepr
from typing import Any, List


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

    Example::

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


def make_bold(x, disable=False):
    """Make input bold

    :param disable: If *True*, don't apply formatting.
    :param x: string input
    """
    if disable:
        return x
    return f'\33[1m{x}\33[0m'


def make_green(x, disable=False):
    """Make input green

    :param disable: If *True*, don't apply formatting.
    :param x: string input
    """
    if disable:
        return x
    return f'\33[32m{x}\33[0m'


def create_arrow(start_left: int, finish_right: int, n_initial_rows: int, n_final_rows: int) -> str:
    """
    Create an ascii arrow.

    :param start_left: X-coordinate where the last line arrow ends.
    :param finish_right: X-coor where the first line arrow starts.
    :param n_initial_rows: Y-padding.
    :param n_final_rows: Y-padding.

    Example::

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
