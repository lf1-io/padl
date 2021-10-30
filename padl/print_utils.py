from reprlib import repr


def combine_multi_line_strings(strings):
    """
    Combine multi line strings, where every character takes precedence over space.
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


def create_reverse_arrow(start_left, finish_right, n_initial_rows, n_final_rows):
    """
    Create an ascii arrow.

    :param start_left: x-coor where last line arrow ends
    :param finish_right: x-coor where first line arrow starts
    :param n_initial_rows: y-padding
    :param n_final_rows: y-padding

    Example:

        ____________|
        |
        v
    """
    final = ''
    for _ in range(n_initial_rows - 1):
        final += ' ' * start_left + '|' + '\n'
    initial = ''
    for _ in range(n_final_rows - 1):
        initial += ' ' * (start_left + finish_right) + '|' + '\n'
    underscore_line = list('_' * finish_right)
    if underscore_line:
        underscore_line[0] = ' '
    underscore_line = ''.join(underscore_line)
    output = initial \
           + ' ' * start_left + underscore_line + '|' + ' ' + '\n' \
           + final \
           + ' ' * start_left + '▼'
    output = '\n'.join(output.split('\n')[1:])
    return output


def make_bold(x):
    """Make input bold

    :param x: string input
    """
    return f'\33[1m{x}\33[0m'


def make_faint(x):
    """Make input faint

    :param x: string input
    """
    return f'\33[2m{x}\33[0m'


def make_green(x):
    """Make input green

    :param x: string input
    """
    return f'\33[32m{x}\33[0m'


def create_arrow(start_left, finish_right, n_initial_rows, n_final_rows):
    """
    Create an ascii arrow.

    :param start_left: x-coor where first line arrow starts
    :param finish_right: x-coor where last line arrow ends
    :param n_initial_rows: y-padding
    :param n_final_rows: y-padding

    Example:

        ____________|
        |
        v
    """
    initial = ''
    for _ in range(n_initial_rows - 1):
        initial += ' ' * start_left + '|' + '\n'
    final = ''
    for _ in range(n_final_rows - 1):
        final += ' ' * (start_left + finish_right) + '|' + '\n'
    return initial + ' ' * start_left + '|' \
        + '_' * (finish_right - 1) + '\n' \
        + final \
        + ' ' * (start_left + finish_right) + '▼'


def format_argument(value):
    """Format an argument value for printing."""
    return repr(value)
