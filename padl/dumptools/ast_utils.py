import ast
from dataclasses import dataclass

NEW_AST_FEATURES = hasattr(ast, 'get_source_segment')

if not NEW_AST_FEATURES:
    from asttokens import ASTTokens, LineNumbers
    import hashlib


TREECACHE = {}


@dataclass
class Position:
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int


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
