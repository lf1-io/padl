import ast
import builtins
from collections import Counter, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Set

from padl.dumptools import ast_utils
from padl.dumptools.symfinder import find_in_scope, ScopedName, Scope

try:
    unparse = ast.unparse
except AttributeError:  # python < 3.9
    from astunparse import unparse


class Finder(ast.NodeVisitor):
    """:class:`ast.NodeVisitor` for finding AST-nodes of a given type in an AST-tree.

    Example:

    >>> Finder(ast.Name).find(ast.parse('x(y)'))  # doctest: +ELLIPSIS
    [<_ast.Name object at 0x...>, <_ast.Name object at 0x...>]
    """

    def __init__(self, nodetype):
        self.nodetype = nodetype
        self.result = []

    def generic_visit(self, node):
        if isinstance(node, self.nodetype):
            self.result.append(node)
        for child_node in ast.iter_child_nodes(node):
            child_node.parent = node
        super().generic_visit(node)

    def find(self, node):
        """Find sub-nodes of *node*. """
        self.visit(node)
        return self.result

    def get_source_segments(self, source):
        """Get a list of source segments of found nodes in source and their respective positions.

        :return: A list of tuples (<source segment>, <position>) where position is a tuple
            (<lineno>, <end lineno>, <col offset>, <end col offset>).

        Example:

        >>> Finder(ast.Name).get_source_segments('x(y)')
        [('x', Position(lineno=1, end_lineno=1, col_offset=0, end_col_offset=1)), ('y', Position(lineno=1, end_lineno=1, col_offset=2, end_col_offset=3))]
        """
        nodes = self.find(ast.parse(source))
        return [
            (
                ast_utils.get_source_segment(source, node),
                ast_utils.get_position(source, node)
            )
            for node in nodes
        ]


def _join_attr(node):
    if not isinstance(node, (ast.Attribute, ast.Name)):
        raise TypeError()
    try:
        return [node.id]
    except AttributeError:
        return _join_attr(node.value) + [node.attr]


Vars = namedtuple('Vars', 'globals locals')


class _VarFinder(ast.NodeVisitor):
    # pylint: disable=invalid-name
    """An :class:`ast.NodeVisitor` that traverses all subnodes of an AST-node and finds all named
    things (variables, functions, classes, modules etc) used in it.

    The results are grouped as "globals" and "locals". "locals" are those things whose definition
    is found under the node itself whereas "globals" are all others.

    >>> source = '''
    ... def f(x):
    ...     y = a + 1
    ...     z = np.array(x + b)
    ...     return str(z)
    ... '''
    >>> _VarFinder().find_in_source(source) == Vars(globals={ScopedName('str', None, 0),
    ...                                                      ScopedName('np.array', None, 0),
    ...                                                      ScopedName('a', None, 0),
    ...                                                      ScopedName('b', None, 0)},
    ...                                             locals={ScopedName('x', None, 0),
    ...                                                     ScopedName('z', None, 0),
    ...                                                     ScopedName('y', None, 0)})
    True
    """

    def __init__(self):
        super().__init__()
        self.globals = set()
        self.locals = set()

    def find(self, node):
        """Find all globals and locals in an AST-node.

        :param node: An ast node to search.
        :returns: Tuple of sets with names of globals and locals.
        """
        if isinstance(node, list):
            for node_ in node:
                self.find(node_)
            return Vars(self.globals, self.locals)
        if isinstance(node, ast.FunctionDef):
            return self._find_in_function_def(node)
        self.visit(node)
        return Vars(self.globals, self.locals)

    def find_in_source(self, source):
        """Find all globals and locals in a piece of source code."""
        return self.find(ast.parse(source).body[0])

    def _find_in_function_def(self, node):
        """This is a special case: Functions args are "locals" rather than "globals".

        Example:

        >>> source = '''
        ... def f(x):
        ...     y = a + 1
        ...     z = np.array(x + b)
        ...     return str(z)
        ... '''
        >>> _VarFinder().find_in_source(source) == Vars(globals={ScopedName('str', None, 0),
        ...                                                      ScopedName('np.array', None, 0),
        ...                                                      ScopedName('a', None, 0),
        ...                                                      ScopedName('b', None, 0)},
        ...                                             locals={ScopedName('x', None, 0),
        ...                                                     ScopedName('z', None, 0),
        ...                                                     ScopedName('y', None, 0)})
        True
        """
        posonlyargs = getattr(node.args, 'posonlyargs', [])
        for arg in node.args.args + posonlyargs + node.args.kwonlyargs:
            self.locals.add(ScopedName(arg.arg, None, 0))
        if node.args.vararg is not None:
            self.locals.add(ScopedName(node.args.vararg.arg, None, 0))
        if node.args.kwarg is not None:
            self.locals.add(ScopedName(node.args.kwarg.arg, None, 0))
        for n in ast.iter_child_nodes(node):
            self.visit(n)
        return Vars(self.globals, self.locals)

    def visit_Name(self, node):
        """Names - Every `Name`'s id is a global unless it's a local.

        Example:

        >>> _VarFinder().find_in_source('x')
        Vars(globals={ScopedName(name='x', scope=None, n=0)}, locals=set())
        """
        scope = getattr(node, '_scope', None)
        name = ScopedName(node.id, scope, 0)
        if not self.in_locals(name):
            self.globals.add(name)

    def visit_Attribute(self, node):
        """Names - Every `Name`'s id is a global unless it's a local.

        Example:

        >>> _VarFinder().find_in_source('x')
        Vars(globals={ScopedName(name='x', scope=None, n=0)}, locals=set())
        """
        try:
            path = _join_attr(node)
        except TypeError:
            self.generic_visit(node)
            return

        scope = getattr(node, '_scope', None)
        name = ScopedName('.'.join(path), scope, 0)
        if self.in_locals(name):
            return
        self.globals.add(name)

    def in_locals(self, name):
        return self.in_ignoring_attributes(name, self.locals)

    def in_ignoring_attributes(self, name, name_set):
        if name in name_set:
            return True
        if '.' not in name.name:
            return False

        path = name.name.split('.')
        for i in range(len(path), 0, -1):
            subname = '.'.join(path[:i])
            if ScopedName(subname, name.scope, 0) in name_set:
                return True
        return False

    def visit_withitem(self, node):
        """With statements - The "as ..." of a with statement is a local.

        Example:

        >>> source = '''
        ... with open('file') as f:
        ...     ...
        ... '''
        >>> _VarFinder().find_in_source(source)
        Vars(globals={ScopedName(name='open', scope=None, n=0)}, locals={ScopedName(name='f', scope=None, n=0)})
        """
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.locals.add(ScopedName(node.optional_vars.id, None, 0))

    def visit_Assign(self, node):
        """Assignments - Their targets are locals, their values are globals.

        Example:

        >>> _VarFinder().find_in_source('x = y')
        Vars(globals={ScopedName(name='y', scope=None, n=0)}, locals={ScopedName(name='x', scope=None, n=0)})
        """
        # collect targets (the 'x' in 'x = a', can be multiple due to 'x = y = a')
        targets = set()
        for target in node.targets:
            # exclude assignment to subscript ('x[1] = a')
            if isinstance(target, ast.Subscript):
                continue
            # exclude assignment to attribute ('x.y = a')
            if isinstance(target, ast.Attribute):
                continue
            scope = getattr(target, '_scope', None)
            targets.update(
                {ScopedName(x.id, scope, 0) for x in Finder(ast.Name).find(target)}
            )
        # find globals in RHS
        sub_globals = {name for name in find_globals(node.value) if not self.in_locals(name)}
        sub_dependencies = set()
        # if a variable on the RHS is one of the targets, increase its counter
        for name in sub_globals:
            if self.in_locals(name):
                continue
            if name in targets:
                sub_dependencies.add(ScopedName(name.name, name.scope, name.n + 1))
            else:
                sub_dependencies.add(name)
        self.locals.update(targets)
        self.globals.update(sub_dependencies)

    def visit_For(self, node):
        """For loops - Looped over items are locals.

        Example:

        >>> source = '''
        ... for x in range(10):
        ...     ...
        ... '''
        >>> _VarFinder().find_in_source(source)
        Vars(globals={ScopedName(name='range', scope=None, n=0)}, locals={ScopedName(name='x', scope=None, n=0)})
        """
        self.locals.update([ScopedName(x.id, getattr(x, '_scope', None), 0)
                            for x in Finder(ast.Name).find(node.target)])
        for child in node.body:
            self.visit(child)
        self.visit(node.iter)

    def visit_NamedExpr(self, node):
        """The walrus operator - it's targets become locals.

        Example:

        >>> source = '''
        ... while a := l.pop():
        ...     ...
        ... '''
        >>> _VarFinder().find_in_source(source)
        Vars(globals={ScopedName(name='l.pop', scope=None, n=0)}, locals={ScopedName(name='a', scope=None, n=0)})
        """
        self.locals.update([ScopedName(x.id, getattr(x, '_scope', None), 0)
                            for x in Finder(ast.Name).find(node.target)])
        self.visit(node.value)

    def handle_comprehension(self, node):
        """Comprehensions are a special case: Their targets should be ignored. """
        targets = set()
        for gen in node.generators:
            for name in Finder(ast.Name).find(gen.target):
                targets.add(ScopedName(name.id, getattr(name, '_scope', None), 0))
        sub_globals = set.union(*[find_globals(n) for n in ast.iter_child_nodes(node)])
        sub_globals = {n for n in sub_globals
                       if not self.in_locals(n)
                       and not self.in_ignoring_attributes(n, targets)}
        self.globals.update(sub_globals)

    def visit_DictComp(self, node):
        """Dict comprehensions.

        Example:

        >>> _VarFinder().find_in_source('{k: v for k, v in foo}')
        Vars(globals={ScopedName(name='foo', scope=None, n=0)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_ListComp(self, node):
        """List comprehensions.

        Example:

        >>> _VarFinder().find_in_source('[x for x in foo]')
        Vars(globals={ScopedName(name='foo', scope=None, n=0)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_SetComp(self, node):
        """Set comprehensions.

        Example:

        >>> _VarFinder().find_in_source('{x for x in foo}')
        Vars(globals={ScopedName(name='foo', scope=None, n=0)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_GeneratorExp(self, node):
        """Generator expressions.

        Example:

        >>> _VarFinder().find_in_source('(x for x in foo)')
        Vars(globals={ScopedName(name='foo', scope=None, n=0)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_Lambda(self, node):
        """Lambda expressions - Their arguments are locals.

        Example:

        >>> vars = _VarFinder().find_in_source('lambda x, y: x + y + foo')
        >>> vars == Vars(globals={ScopedName('foo', None, 0)}, locals={ScopedName('x', None, 0),
        ...                                                            ScopedName('y', None, 0)})
        True
        """
        scope = getattr(node, '_scope', None)
        posonlyargs = getattr(node.args, 'posonlyargs', [])
        for arg in node.args.args + posonlyargs + node.args.kwonlyargs:
            self.locals.add(ScopedName(arg.arg, scope, 0))
        self.visit(node.body)

    def visit_FunctionDef(self, node):
        """Function definitions. """
        scope = getattr(node, '_scope', None)
        self.locals.add(ScopedName(node.name, scope, 0))
        inner_globals = find_globals(node)
        self.globals.update(v for v in inner_globals if not self.in_locals(v))

    def visit_ClassDef(self, node):
        """Class definitions. """
        scope = getattr(node, '_scope', None)
        self.locals.add(ScopedName(node.name, scope, 0))
        inner_globals = find_globals(node)
        self.globals.update(v for v in inner_globals if not self.in_locals(v))

    def visit_Import(self, node):
        """Import statements.

        Example:

        >>> _VarFinder().find_in_source('import foo')
        Vars(globals=set(), locals={ScopedName(name='foo', scope=None, n=0)})
        >>> _VarFinder().find_in_source('import foo as bar')
        Vars(globals=set(), locals={ScopedName(name='bar', scope=None, n=0)})
        """
        scope = getattr(node, '_scope', None)
        for name in node.names:
            if name.asname is None:
                self.locals.add(ScopedName(name.name, scope, 0))
            else:
                self.locals.add(ScopedName(name.asname, scope, 0))

    def visit_ImportFrom(self, node):
        """Import-from statements.

        Example:

        >>> _VarFinder().find_in_source('from foo import bar')
        Vars(globals=set(), locals={ScopedName(name='bar', scope=None, n=0)})
        >>> _VarFinder().find_in_source('from foo import bar as baz')
        Vars(globals=set(), locals={ScopedName(name='baz', scope=None, n=0)})
        """
        scope = getattr(node, '_scope', None)
        for name in node.names:
            if name.asname is None:
                self.locals.add(ScopedName(name.name, scope, 0))
            else:
                self.locals.add(ScopedName(name.asname, scope, 0))


class _Renamer(ast.NodeTransformer):
    # pylint: disable=invalid-name
    """An :class:`ast.NodeTransformer` for renaming things. Used in :func:`rename`.

    :param from_: Rename everything called this ...
    :param to: ... to this.
    :param rename_locals: If *True*, rename local things, else, only rename globals.
    """

    def __init__(self, from_, to, rename_locals):
        self.from_ = from_
        self.to = to
        self.rename_locals = rename_locals

    def visit_arg(self, node):
        """Arguments.

        Example:

        >>> node = ast.parse('''
        ... def f(a, b):
        ...    ...
        ... ''').body[0]
        >>> unparse(rename(node, 'a', 'c', rename_locals=True))
        '\\n\\ndef f(c, b):\\n    ...\\n'
        """
        if node.arg == self.from_:
            return ast.arg(**{**node.__dict__, 'arg': self.to})
        return node

    def visit_Name(self, node):
        """Names.

        Example:

        >>> node = ast.parse('x')
        >>> unparse(rename(node, 'x', 'y', True)).strip()
        'y'
        """
        if node.id == self.from_:
            return ast.Name(**{**node.__dict__, 'id': self.to})
        return node

    def visit_FunctionDef(self, node):
        """Function definitions.

        Example:

        >>> node = ast.parse('''
        ... def f(a, b):
        ...    y = p(z)
        ... ''').body[0]
        >>> unparse(rename(node, 'z', 'u', rename_locals=True))
        '\\n\\ndef f(a, b):\\n    y = p(u)\\n'
        """
        if self.rename_locals and node.name == self.from_:
            name = self.to
        else:
            name = node.name
        if not self.rename_locals:
            globals_ = find_globals(node)
            if self.from_ not in {x.name for x in globals_}:
                return node
        return ast.FunctionDef(**{**node.__dict__,
                                  'args': rename(node.args, self.from_, self.to,
                                                 self.rename_locals),
                                  'body': rename(node.body, self.from_, self.to,
                                                 self.rename_locals),
                                  'name': name})

    def visit_ClassDef(self, node):
        """Class definitions.

        Example:

        >>> node = ast.parse('''
        ... class Foo:
        ...     def __init__(self, a, b):
        ...         y = p(z)
        ... ''').body[0]
        >>> unparse(rename(node, 'z', 'u', rename_locals=True))
        '\\n\\nclass Foo():\\n\\n    def __init__(self, a, b):\\n        y = p(u)\\n'
        """
        if self.rename_locals and node.name == self.from_:
            name = self.to
        else:
            name = node.name
        if not self.rename_locals:
            globals_ = find_globals(node)
            if self.from_ not in {x.name for x in globals_}:
                return node
        return ast.ClassDef(**{**node.__dict__,
                               'body': rename(node.body, self.from_, self.to,
                                              self.rename_locals),
                               'name': name})


def rename(tree: ast.AST, from_: str, to: str, rename_locals: bool = False):
    """Rename things in an AST tree.

    :param tree: The tree in which to rename things.
    :param from_: Rename everything called this ...
    :param to: ... to this.
    :param rename_locals: If *True*, rename local things, else, only rename globals.
    """
    if not rename_locals:
        globals_ = find_globals(tree)
        if from_ not in {x.name for x in globals_}:
            return tree
    if isinstance(tree, Iterable) and not isinstance(tree, str):
        return [rename(x, from_, to, rename_locals) for x in tree]
    if not isinstance(tree, ast.AST):
        return tree
    renamer = _Renamer(from_, to, rename_locals)
    return renamer.visit(tree)


class _MethodFinder(ast.NodeVisitor):
    """Find all methods in a class node.

    Example:

    >>> source = '''
    ... class X:
    ...     def one(self):
    ...         ...
    ...
    ...     def two(self):
    ...         ...
    ... '''
    >>> node = ast.parse(source).body[0]
    >>> _MethodFinder().find(node)  # doctest: +ELLIPSIS
    {'one': <_ast.FunctionDef object at 0x...>, 'two': <_ast.FunctionDef object at 0x...>}
    """

    def __init__(self):
        self.methods = {}

    def visit_FunctionDef(self, node):
        self.methods[node.name] = node

    def find(self, node: ast.ClassDef) -> Dict['str', ast.FunctionDef]:
        """Do find the methods. """
        self.visit(node)
        return self.methods


def _filter_builtins(names):
    return {name for name in names if name.name not in builtins.__dict__}

def _find_globals_in_classdef(node: ast.ClassDef, filter_builtins: bool = True):
    """Find globals used below a ClassDef node. """
    methods = _MethodFinder().find(node)
    # globals used in class body
    globals_ = set()
    for child in ast.iter_child_nodes(node):
        globals_.update(_VarFinder().find(child).globals)
    # globals used in methods
    for method in methods.values():
        globals_.update(_VarFinder().find(method).globals)
    if filter_builtins:
        globals_ = _filter_builtins(globals_)
    return globals_


def find_globals(node: ast.AST, filter_builtins: bool = True) -> Set[Tuple[str, int]]:
    """Find all globals used below a node.

    Example:

    >>> source = '''
    ... def f(x):
    ...     y = a + 1
    ...     z = np.array(x + b)
    ...     return str(z)
    ... '''
    >>> node = ast.parse(source).body[0]
    >>> find_globals(node) == {ScopedName('a', None, 0), ScopedName('b', None, 0),
    ...                        ScopedName('np.array', None, 0)}
    True

    :param node: AST node to search in.
    :param filter_builtins: If *True*, filter out builtins.
    """
    if isinstance(node, ast.ClassDef):
        return _find_globals_in_classdef(node)
    globals_ = _VarFinder().find(node).globals
    if filter_builtins:
        globals_ = _filter_builtins(globals_)
    return globals_


def increment_same_name_var(variables: List[ScopedName], scoped_name: ScopedName):
    """Go through *variables* and increment the the counter for those with the same name as
    *scoped_name* by *scoped_name.n*.

    Example:

    >>> import padl as somemodule
    >>> out = increment_same_name_var({ScopedName('a', None, 1), ScopedName('b', None, 2)},
    ...                               ScopedName('b', somemodule, 2))
    >>> isinstance(out, set)
    True
    >>> {(x.name, x.n) for x in out} == {('a', 1), ('b', 4)}
    True
    """
    result = set()
    for var in variables:
        if var.scope is None:
            scope = scoped_name.scope
        else:
            scope = var.scope

        if var.name == scoped_name.name:
            result.add(ScopedName(var.name, scope, var.n + scoped_name.n))
        else:
            result.add(ScopedName(var.name, scope, var.n))
    return result


def find_codenode(name: ScopedName, full_dump_module_names=None):
    """Find the :class:`CodeNode` corresponding to a :class:`ScopedName` *name*. """
    (source, node), scope_of_next_var, found_name = find_in_scope(name)

    module_name = None
    if full_dump_module_names:
        if isinstance(node, ast.Import):
            module_name = node.names[0].name
        if isinstance(node, ast.ImportFrom):
            module_name = node.module
    if module_name is not None:
        if any(module_name.startswith(mod) for mod in full_dump_module_names):
            return find_codenode(ScopedName(name.name, Scope.toplevel(module_name), name.n),
                                 full_dump_module_names)

    # find dependencies
    globals_ = find_globals(node)
    next_name = ScopedName(name.name, scope_of_next_var, name.n)
    globals_ = increment_same_name_var(globals_, next_name)

    return CodeNode(source=source, globals_=globals_, ast_node=node, scope=scope_of_next_var,
                    name=found_name, n=name.n)


def _get_nodes_without_in_edges(graph):
    """Get all nodes in directed graph *graph* that don't have incoming edges.

    The graph is represented by a dict mapping nodes to incoming edges.

    Example:

    >>> graph = {'a': [], 'b': ['a'], 'c': ['a'], 'd': ['b']}
    >>> _get_nodes_without_in_edges(graph)
    ({'a'}, {'b': set(), 'c': set(), 'd': {'b'}})

    :param graph: A dict mapping nodes to incoming edges.
    :return: The set of nodes without incoming edges and the graph with these nodes removed.
    """
    nextlevel = set()
    for node, deps in graph.items():
        if not deps or deps == {node}:
            nextlevel.add(node)
    filtered_graph = {}
    for node, deps in graph.items():
        if node in nextlevel:
            continue
        filtered_graph[node] = \
            {dep for dep in deps if dep not in nextlevel}
    return nextlevel, filtered_graph


# sort precedence
_PRECEDENCE = {
    # imports and modules on top, ordered by module name
    ast.Import: lambda k, v: '1' + v.ast_node.names[0].name.lower() + k.lower(),
    ast.ImportFrom: lambda k, v: '1' + v.ast_node.module.lower() + k.lower(),
    # assignments (-> constants) after that
    ast.Assign: lambda k, _v: '2' + k.lower()
    # the rest goes below
}


def _topsort(graph: dict) -> List[set]:
    """Topologically sort a graph represented by a dict mapping nodes to incoming edges.

    Raises a :exc:`RuntimeError` if the graph contains a cycle.

    Example:

    >>> graph = {'a': [], 'b': ['a'], 'c': ['a'], 'd': ['b']}
    >>> _topsort(graph) == [{'a'}, {'b', 'c'}, {'d'}]
    True

    :param graph: Graph represented by a dict mapping nodes to incoming edges.
    :return: List of set where each contained set represents one level, the first level
        has no dependencies, each subsequent level depends on nodes in the previous level.
    """
    levels = []
    graphlen = len(graph)
    while graph:
        nextlevel, graph = _get_nodes_without_in_edges(graph)
        if graphlen == len(graph):  # graph didn't shrink
            raise RuntimeError('Graph has a circle or dangling roots.')
        graphlen = len(graph)
        levels.append(nextlevel)
    return levels


@dataclass
class CodeNode:
    """A node in a :class:`CodeGraph`.

    A `CodeNode` has

    - a *source*: The source code represented by the `CodeNode`.
    - a set of *globals_*: Names the node depends on.
    - (optionally) a *scope*: The module and function scope the `CodeNode` lives in.
    - (optionally) an *ast_node*: An `ast.AST` object representing the code.
    """
    source: str
    globals_: set
    name: str
    scope: Optional[Scope] = None
    ast_node: Optional[ast.AST] = None
    n: int = 0

    @classmethod
    def from_source(cls, source, scope, name):
        """Build a `CodeNode` from a source string. """
        node = ast.parse(source).body[0]
        globals_ = {
            ScopedName(name.name, scope, name.n)
            for name in find_globals(node)
        }

        return cls(
            source=source,
            ast_node=node,
            globals_=globals_,
            scope=scope,
            name=name
        )

    def __hash__(self):
        return hash((self.name, self.scope, self.n))

    def __eq__(self, other):
        return (
            (self.name, self.scope, self.n)
            == (other.name, other.scope, other.n)
        )


def _sort(unscoped_graph):
    top = _topsort({k: v.globals_ for k, v in unscoped_graph.items()})

    def sortkey(x):
        val = unscoped_graph[x]
        try:
            return _PRECEDENCE[val.ast_node.__class__](x[0], val)
        except KeyError:
            return 'zz' + x[0].lower()

    res = []
    for level in top:
        res += sorted(level, key=sortkey)
    return res


def _dumps_unscoped(unscoped_graph):
    """Dump an unscoped (see :meth:`CodeGraph.unscope`) :class:`CodeGraph` to a python source
    string. """
    sorted_ = _sort(unscoped_graph)
    res = ''
    done = set()
    for i, name in enumerate(sorted_):
        here = unscoped_graph[name]
        if here in done:
            continue
        if not here.source.strip():
            continue
        done.add(here)
        res += here.source
        if i < len(sorted_) - 1:
            next_ = unscoped_graph[sorted_[i + 1]]
            if isinstance(here.ast_node, (ast.Import, ast.ImportFrom)) \
                    and isinstance(next_.ast_node, (ast.Import, ast.ImportFrom)):
                res += '\n'
            elif isinstance(here.ast_node, ast.Assign) \
                    and isinstance(next_.ast_node, ast.Assign):
                res += '\n'
            else:
                res += '\n\n\n'
    return res + '\n'


class CodeGraph(dict):
    """A graph representing python code.

    The nodes in the graph are `CodeNode` objects, representing pieces of code defining python
    variables. The edges are the dependencies between the nodes.

    As an example - the following code::

        import foo

        def f(x):
            return foo.bar(x) + 1

        o = f(100)

    defines a codegraph with the nodes::

        foo: "import foo"
        f: "def f(x) ..."
        o: "o = f(100)"

    Edges are determined by the variable dependencies between the nodes::

        o -> f -> foo

    As `o` depends on `f` and `f` depends on `foo`.
    """

    def _unscoped(self):
        """Create a version of *self* where all non-top level variables are renamed (by prepending
        the scope) to prevent conflicts."""
        name_scope = {(k.name, v.scope) for k, v in self.items()}
        counts = Counter(x[0] for x in name_scope)
        to_rename = set(k for k, c in counts.items() if c > 1)

        def unscope(name, scope):
            if name in to_rename:
                return scope.unscoped(name)
            return name

        res = {}
        for k, v in self.items():
            changed = False
            k_unscoped = unscope(k.name, k.scope)
            v_unscoped = unscope(v.name, k.scope)
            changed = changed or k_unscoped != k.name
            code = v.source
            tree = ast.parse(code)
            rename(tree, k.name, k_unscoped, rename_locals=True)
            vars_ = set()
            for var in list(v.globals_):
                var_unscoped = unscope(var.name, var.scope)
                changed = changed or var_unscoped != var.name
                rename(tree, var.name, var_unscoped)
                vars_.add((var_unscoped, var.n))
            if changed:
                code = unparse(tree).strip('\n')
            res[k_unscoped, k.n] = CodeNode(code, vars_, ast_node=v.ast_node, name=v_unscoped,
                                            n=v.n)
        return res

    def dumps(self):
        """Create a python source string with the contents of the graph. """
        return _dumps_unscoped(self._unscoped())

    @classmethod
    def build(cls, scoped_name: ScopedName):
        """Build a codegraph corresponding to a `ScopedName`.

        The name will be searched for in its scope.
        """
        graph = cls()
        done = set()

        todo = {scoped_name}

        while todo:
            # we know this already - go on
            next_name = todo.pop()

            if next_name in done:
                continue

            # find how next_var came into being
            next_codenode = find_codenode(next_name)
            graph[next_name] = next_codenode

            todo.update(next_codenode.globals_)
            done.add(next_name)

        return graph

    def print(self):
        """Print the graph (for debugging). """
        for k, v in self.items():
            print(f'{k.name} {k.scope} {k.n}:')
            print()
            print(v.source)
            print()
            for dep in v.globals_:
                print(f'{dep.name} {dep.scope} {dep.n}:')
            print()
            print('--------')
            print()
