import inspect
import linecache
from typing import List


replace_cache = {}


def get_source(filename: str, use_replace_cache=True):
    """Get source from *filename*.

    The filename is as in the code object gotten from an `inspect.frame`,
    this can be "<ipython input-...>" in which case the source is taken from the ipython cache.

    If *use_replace_cache*, try getting the source from the "replace_cache", which can contain
    explicit replacements of the original source strings.
    """
    if use_replace_cache and filename in replace_cache:
        return replace_cache[filename]
    if filename in linecache.cache:
        # the ipython case
        return ''.join(linecache.cache[filename][2])
    # normal module
    with open(filename) as f:
        return f.read()


def get_module_source(module, use_replace_cache=True):
    """Get the source code of a module.

    If *use_replace_cache*, try getting the source from the "replace_cache", which can contain
    explicit replacements of the original source strings.
    """
    if use_replace_cache:
        try:
            return replace_cache[module.__filename__]
        except (KeyError, AttributeError):
            pass
    return inspect.getsource(module)


def _ipython_history():
    """Get the list of commands executed by IPython, ordered from oldest to newest. """
    return [
        replace_cache.get(k, ''.join(lines))
        for k, (_, _, lines, _)
        in linecache.cache.items()
        if k.startswith('<ipython-')
        or 'ipykernel' in k
    ]


def original(string: str):
    """Get either the original of a `ReplaceString` or a string."""
    return getattr(string, 'original', string)


class ReplaceString(str):
    """A string with a replaced section.

    Has an attribute `original` which gives the original string (pre-replacing).

    :param string: The original string.
    :param what: The string to insert at the specified location.
    :param from_line: The first line to replace in.
    :param to_line: The last line to replace in.
    :param from_col: The first col on *from_line* to replace.
    :param to_col: The last col on *to_line* to replace.
    """

    def __new__(cls, string, what, from_line, to_line, from_col, to_col):
        replaced = replace(string, what, from_line, to_line, from_col, to_col)
        return super().__new__(cls, replaced)

    def __init__(self, string, what, from_line, to_line, from_col, to_col):
        self.original = string
        self.from_line = from_line
        self.to_line = to_line
        self.from_col = from_col
        self.to_col = to_col
        self.what = what
        super().__init__()

    def cut(self, from_line, to_line, from_col, to_col):
        """Cut and return the resulting sub-`ReplaceString`. """
        lines = self.original.split('\n')[from_line: to_line + 1]
        lines[0] = lines[0][from_col:]
        lines[-1] = lines[-1][:to_col]

        new_from_line = self.from_line - from_line
        new_to_line = self.to_line - from_line
        if from_line == self.from_line:
            new_from_col = self.from_col - from_col
        else:
            new_from_col = self.from_col
        if to_line == self.to_line and to_col < self.to_col:
            new_to_col = to_col
        else:
            new_to_col = self.to_col
        if self.to_line == from_line:
            new_to_col -= from_col

        if new_from_line < 0:
            new_from_col = 0
        if new_to_line < 0:
            new_from_col = 0
            new_to_col = 0

        return ReplaceString('\n'.join(lines), self.what, new_from_line, new_to_line, new_from_col,
                             new_to_col)


class ReplaceStrings(str):
    """A collection of replacestrings with different replacements in an original string. """

    def __new__(cls, rstrings: List[ReplaceString]):
        rstrings = sorted(rstrings, key=lambda x: (x.from_line, x.from_col), reverse=True)
        replaced = rstrings[0].original
        for rstring in rstrings:
            replaced = replace(replaced, rstring.what, rstring.from_line, rstring.to_line,
                               rstring.from_col, rstring.to_col)
        return super().__new__(cls, replaced)

    def __init__(self, rstrings):
        super().__init__()
        self.original = rstrings[0].original
        self.rstrings = rstrings

    def cut(self, from_line, to_line, from_col, to_col):
        """Cut and return the resulting sub-`ReplaceStrings`. """
        return ReplaceStrings([rstr.cut(from_line, to_line, from_col, to_col)
                               for rstr in self.rstrings])


def put_into_cache(key, source: str, repl: str, *loc):
    val = ReplaceString(source, repl, *loc)
    try:
        replace_cache[key] = ReplaceStrings([val] + replace_cache[key].rstrings)
    except KeyError:
        replace_cache[key] = ReplaceStrings([val])


def cut_string(string: str, from_line: int, to_line: int, from_col: int, to_col: int):
    """Cut a string and return the resulting substring.

    Example:
        Given a *string*,

        "xxxxxxxxxxx
         xxxxxAXXXXXXXXXXXXXXXXXXXX
         XXXXXXXXXXXXXBxxxx
         xxxx"

        , to cut a substring from *A* to *B*, give as *from_line* the line of *A* (1) and *to_line*
        the line of *B* (2). *from_col* determines the position of *A* within the line (5) and
        *to_col* determines the position of *B* (13).
        The result would be:

        "AXXXXXXXXXXXXXXXXXXXX
         XXXXXXXXXXXXXB"

    :param string: The input string.
    :param from_line: The first line to include.
    :param to_line: The last line to include.
    :param from_col: The first col on *from_line* to include.
    :param to_col: The last col on *to_line* to include.
    :returns: The cut-out string.
    """
    lines = string.split('\n')[from_line: to_line + 1]
    if len(lines) == 1:
        return lines[0][from_col:to_col]
    lines[0] = lines[0][from_col:]
    lines[-1] = lines[-1][:to_col]
    return '\n'.join(lines)


def cut(string: str, from_line: int, to_line: int, from_col: int, to_col: int):
    """Cut a string (can be a normal `str`, a `ReplaceString` or a `ReplaceStrings`) and return the
    resulting substring.

    Example:

        Given a *string*,

        "xxxxxxxxxxx
         xxxxxAXXXXXXXXXXXXXXXXXXXX
         XXXXXXXXXXXXXBxxxx
         xxxx"

        , to cut a substring from *A* to *B*, give as *from_line* the line of *A* (1) and *to_line*
        the line of *B* (2). *from_col* determines the position of *A* within the line (5) and
        *to_col* determines the position of *B* (13).
        The result would be:

        "AXXXXXXXXXXXXXXXXXXXX
         XXXXXXXXXXXXXB"

    If used with a `ReplaceString`, the result will be a `ReplaceString` with original and
    replacement at the expected positions.

    :param string: The input string.
    :param from_line: The first line to include.
    :param to_line: The last line to include.
    :param from_col: The first col on *from_line* to include.
    :param to_col: The last col on *to_line* to include.
    :returns: The cut-out string.
    """
    try:
        return string.cut(from_line, to_line, from_col, to_col)
    except AttributeError:
        return cut_string(string, from_line, to_line, from_col, to_col)


def replace(string, what, from_line, to_line, from_col, to_col):
    """Replace a substring in *string* with what. """
    if from_line < 0 and to_line < 0:
        return string

    lines = string.split('\n')

    if from_line > len(lines) - 1 and to_line > len(lines) - 1:
        return string

    if from_line < 0:
        from_col = 0
    from_line = max(from_line, 0)
    if to_line > len(lines) - 1:
        to_col = len(lines[-1])
    to_line = min(to_line, len(lines) - 1)

    for i, line in enumerate(lines[:-1]):
        lines[i] = line + '\n'

    before = ''.join(lines[:from_line])
    if from_line == to_line:
        middle = lines[from_line][:from_col] + what + lines[from_line][to_col:]
        start = end = ''
    else:
        start = lines[from_line][from_col:] + what
        middle = ''.join(lines[from_line+1:to_line])
        end = lines[to_line][:to_col]
    after = ''.join(lines[to_line+1:])

    return before + start + middle + end + after
