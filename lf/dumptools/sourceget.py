import inspect
import linecache


cache = {}


def get_source(filename: str, use_cache=True):
    """Get source from *filename*.

    Filename as in the code object, can be "<ipython input-...>" in which case
    the source is taken from the ipython cache.
    """
    if use_cache and filename in cache:
        return cache[filename]
    if filename in linecache.cache:
        # the ipython case
        return ''.join(linecache.cache[filename][2])
    # normal module
    with open(filename) as f:
        return f.read()


def get_module_source(module, use_cache=True):
    """Get the source code of a module. """
    if use_cache:
        try:
            return cache[module.__filename__]
        except (KeyError, AttributeError):
            pass
    inspect.getsource(module)


def _ipython_history():
    """Get the list of commands executed by IPython, ordered from oldest to newest. """
    return [
        cache.get(k, ''.join(lines))
        for k, (_, _, lines, _)
        in linecache.cache.items()
        if k.startswith('<ipython-')
        or 'ipykernel' in k
    ]


def original(string: str):
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
            new_from_col = max(self.from_col - from_col, 0)
        else:
            new_from_col = self.from_col
        if to_line == self.to_line and to_col < self.to_col:
            new_to_col = to_col
        else:
            new_to_col = self.to_col
        if self.to_line == from_line:
            new_to_col -= from_col

        return ReplaceString('\n'.join(lines), self.what, new_from_line, new_to_line, new_from_col,
                             new_to_col)


class ReplaceStrings(str):
    def __new__(cls, rstrings):
        rstrings = sorted(rstrings, key=lambda x: (x.from_line, x.from_col), reverse=True)
        replaced = rstrings[0].original
        for rstring in rstrings:
            replaced = replace(replaced, rstring.what, rstring.from_line, rstring.to_line,
                               rstring.from_col, rstring.to_col)
        return super().__new__(cls, replaced)

    def __init__(self, rstrings):
        self.original = rstrings[0].original
        self.rstrings = rstrings

    def cut(self, from_line, to_line, from_col, to_col):
        return ReplaceStrings([rstr.cut(from_line, to_line, from_col, to_col)
                               for rstr in self.rstrings])


def put_into_cache(key, source: ReplaceString, repl: str, *loc):
    val = ReplaceString(source, repl, *loc)
    try:
        cache[key] = ReplaceStrings([val] + cache[key].rstrings)
    except KeyError:
        cache[key] = ReplaceStrings([val])




def cut_string(string, from_line, to_line, from_col, to_col):
    """Cut a string and return the resulting substring.

    :param string: The input string.
    :param from_line: The first line to include.
    :param to_line: The last line to include.
    :param from_col: The first col on *from_line* to include.
    :param to_col: The last col on *to_line* to include.
    """
    lines = string.split('\n')[from_line: to_line + 1]
    lines[0] = lines[0][from_col:]
    lines[-1] = lines[-1][:to_col]
    return '\n'.join(lines)


def cut(string: str, from_line: int, to_line: int, from_col: int, to_col: int):
    """Cut a string (can be a `ReplaceString` or a `ReplaceStrings`) and return the resulting
    substring.

    :param string: The input string.
    :param from_line: The first line to include.
    :param to_line: The last line to include.
    :param from_col: The first col on *from_line* to include.
    :param to_col: The last col on *to_line* to include.
    """
    try:
        return string.cut(from_line, to_line, from_col, to_col)
    except AttributeError:
        return cut_string(string, from_line, to_line, from_col, to_col)


def replace(string, what, from_line, to_line, from_col, to_col):
    """Replace a substring in *string* with what. """
    keep_newline = from_line == to_line
    res = ''
    inside = False
    for i, line in enumerate(string.split('\n')):
        if i == from_line:
            startcol = from_col
            inside = True
        else:
            startcol = None
        if i == to_line:
            endcol = to_col
        else:
            endcol = None
        if i > to_line:
            inside = False
        if startcol is not None:
            res += line[:startcol]
        if i == from_line:
            res += what
        if endcol is not None:
            res += line[endcol:]
        if startcol is None and endcol is None:
            res += line
        if not inside or keep_newline:
            res += '\n'
    return res
