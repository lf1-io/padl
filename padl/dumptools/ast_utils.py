import ast
from dataclasses import dataclass
import hashlib
from typing import Tuple

try:
    unparse = ast.unparse
except AttributeError:  # python < 3.9
    from astunparse import unparse


NEW_AST_FEATURES = hasattr(ast, 'get_source_segment')

if not NEW_AST_FEATURES:
    from asttokens import ASTTokens, LineNumbers


TREECACHE = {}
AST_NODE_CACHE = {}


@dataclass
class Position:
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int

    def __iter__(self):
        yield from (self.lineno, self.end_lineno, self.col_offset, self.end_col_offset)


def span_to_pos(span: Tuple[int, int], text: str):
    """Convert a string-*span* (tuple of start- and end- character positions) to a :class:`Position`.
    """
    lineno = text[:span[0]].count('\n') + 1
    end_lineno = text[:span[1]].count('\n') + 1
    col_offset = len(text[:span[0]].split('\n')[-1])
    end_col_offset = len(text[:span[1]].split('\n')[-1])
    return Position(lineno, end_lineno, col_offset, end_col_offset)


def cached_parse(source):
    """Cached version of :func:`ast.parse`.

    If called a second time with the same *source*, this returns a cached tree instead of parsing
    again.
    """
    hash_ = hashlib.md5(source.encode()).digest()
    if hash_ not in AST_NODE_CACHE:
        AST_NODE_CACHE[hash_] = ast.parse(source)
    return AST_NODE_CACHE[hash_]


def get_source_segment(source, node):
    """Get the source segment corresponding to *node*.

    Same as :func:`ast.get_source_segment` in python>=3.8
    """
    if NEW_AST_FEATURES:
        return ast.get_source_segment(source, node)

    # we're caching the trees as asttokens is very slow
    hash_ = hashlib.md5(source.encode()).digest()
    if hash_ in TREECACHE:
        asto = TREECACHE[hash_]
    else:
        asto = ASTTokens(source)
        TREECACHE[hash_] = asto
    asto.mark_tokens(node)
    return asto.get_text(node)


def get_position(source, node):
    """Get the position of *node* in *source*. """
    if NEW_AST_FEATURES:
        return Position(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)

    line_numbers = LineNumbers(source)

    hash_ = hashlib.md5(source.encode()).digest()
    if hash_ in TREECACHE:
        asto = TREECACHE[hash_]
    else:
        asto = ASTTokens(source)
        TREECACHE[hash_] = asto

    asto.mark_tokens(node)
    from_, to = asto.get_text_range(node)
    lineno, col_offset = line_numbers.offset_to_line(from_)
    end_lineno, end_col_offset = line_numbers.offset_to_line(to)
    return Position(lineno, end_lineno, col_offset, end_col_offset)


def dump(node, annotate_fields=True, include_attributes=False, *, indent=2):
    """
    Ast dump with indentation, taken from python 3.9 source.

    Return a formatted dump of the tree in node.  This is mainly useful for
    debugging purposes.  If annotate_fields is true (by default),
    the returned string will show the names and the values for fields.
    If annotate_fields is false, the result string will be more compact by
    omitting unambiguous field names.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    include_attributes can be set to true.  If indent is a non-negative
    integer or string, then the tree will be pretty-printed with that indent
    level. None (the default) selects the single line representation.
    """
    def _format(node, level=0):
        if indent is not None:
            level += 1
            prefix = '\n' + indent * level
            sep = ',\n' + indent * level
        else:
            prefix = ''
            sep = ', '
        if isinstance(node, ast.AST):
            cls = type(node)
            args = []
            allsimple = True
            keywords = annotate_fields
            for name in node._fields:
                try:
                    value = getattr(node, name)
                except AttributeError:
                    keywords = True
                    continue
                if value is None and getattr(cls, name, ...) is None:
                    keywords = True
                    continue
                value, simple = _format(value, level)
                allsimple = allsimple and simple
                if keywords:
                    args.append('%s=%s' % (name, value))
                else:
                    args.append(value)
            if include_attributes and node._attributes:
                for name in node._attributes:
                    try:
                        value = getattr(node, name)
                    except AttributeError:
                        continue
                    if value is None and getattr(cls, name, ...) is None:
                        continue
                    value, simple = _format(value, level)
                    allsimple = allsimple and simple
                    args.append('%s=%s' % (name, value))
            if allsimple and len(args) <= 3:
                return '%s(%s)' % (node.__class__.__name__, ', '.join(args)), not args
            return '%s(%s%s)' % (node.__class__.__name__, prefix, sep.join(args)), False
        elif isinstance(node, list):
            if not node:
                return '[]', True
            return '[%s%s]' % (prefix, sep.join(_format(x, level)[0] for x in node)), False
        return repr(node), True

    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    if indent is not None and not isinstance(indent, str):
        indent = ' ' * indent
    return _format(node)[0]
