"""Module for converting python variables into modules.
"""
import ast
import builtins
from collections import Counter, namedtuple
import copy
from dataclasses import dataclass
import re
from typing import Dict, Optional, List, Tuple, Set, Union

from padl.dumptools import ast_utils
from padl.dumptools import sourceget
from padl.dumptools.symfinder import find_in_scope, Scope, ScopedName


class Finder(ast.NodeVisitor):
    """:class:`ast.NodeVisitor` for finding AST-nodes of a given type in an AST-tree.

    Example:

    >>> Finder(ast.Name).find(ast.parse('x(y)'))  # doctest: +ELLIPSIS
    [<...ast.Name object at 0x...>, <...ast.Name object at 0x...>]
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
        nodes = self.find(ast_utils.cached_parse(source))
        return [
            (
                ast_utils.get_source_segment(source, node),
                ast_utils.get_position(source, node)
            )
            for node in nodes
        ]


def _join_attr(ast_node):
    """Get list of base object name and attribute names.

    if ast_node is a ast.Attribute node with name 'f.pd_to', output is
    ['f', 'pd_to']

    :param ast_node: ast.node
    :return: List of strings representing name of base object and the name of attributes accessed
    """
    if not isinstance(ast_node, (ast.Attribute, ast.Name)):
        raise TypeError()
    try:
        return [ast_node.id]
    except AttributeError:
        return _join_attr(ast_node.value) + [ast_node.attr]


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
    >>> _VarFinder().find_in_source(source) == Vars(globals={ScopedName('str', None),
    ...                                                      ScopedName('np.array', None),
    ...                                                      ScopedName('a', None),
    ...                                                      ScopedName('b', None)},
    ...                                             locals={ScopedName('x', None),
    ...                                                     ScopedName('z', None),
    ...                                                     ScopedName('y', None)})
    True

    Attributes `globals` and `locals` are sets of ScopedName.
    """

    def __init__(self):
        super().__init__()
        self.globals = set()
        self.locals = set()

    def add_to_locals(self, elements):
        self.locals.update(elements)
        self.globals = self.globals - self.locals

    def find(self, node):
        """Find all globals and locals in an AST-node.

        :param node: An ast node to search.
        :returns: Tuple of sets with ScopedName of globals and locals.
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
        return self.find(ast_utils.cached_parse(source).body[0])

    def _find_in_function_def(self, node):
        """This is a special case: Functions args are "locals" rather than "globals".

        Example:

        >>> source = '''
        ... def f(x):
        ...     y = a + 1
        ...     z = np.array(x + b)
        ...     return str(z)
        ... '''
        >>> _VarFinder().find_in_source(source) == Vars(globals={ScopedName('str', None),
        ...                                                      ScopedName('np.array', None),
        ...                                                      ScopedName('a', None),
        ...                                                      ScopedName('b', None)},
        ...                                             locals={ScopedName('x', None),
        ...                                                     ScopedName('z', None),
        ...                                                     ScopedName('y', None)})
        True
        """
        posonlyargs = getattr(node.args, 'posonlyargs', [])
        for arg in node.args.args + posonlyargs + node.args.kwonlyargs:
            self.add_to_locals([ScopedName(arg.arg, None)])
        if node.args.vararg is not None:
            self.add_to_locals([ScopedName(node.args.vararg.arg, None)])
        if node.args.kwarg is not None:
            self.add_to_locals([ScopedName(node.args.kwarg.arg, None)])
        for n in ast.iter_child_nodes(node):
            self.visit(n)
        return Vars(self.globals, self.locals)

    def visit_Name(self, node):
        """Names - Every `Name`'s id is a global unless it's a local.

        Example:

        >>> _VarFinder().find_in_source('x')
        Vars(globals={ScopedName(name='x', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        scope = getattr(node, '_scope', None)
        name = ScopedName(node.id, scope)
        if not self.in_locals(name):
            self.globals.add(name)

    def visit_Attribute(self, node):
        """Names - Every `Name`'s id is a global unless it's a local.

        Example:

        >>> _VarFinder().find_in_source('x')
        Vars(globals={ScopedName(name='x', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        try:
            path = _join_attr(node)
        except TypeError:
            self.generic_visit(node)
            return

        scope = getattr(node, '_scope', None)
        name = ScopedName('.'.join(path), scope)
        if self.in_locals(name):
            return
        self.globals.add(name)

    def in_locals(self, name):
        return self.in_ignoring_attributes(name, self.locals)

    def in_ignoring_attributes(self, name: ScopedName, name_set: Set[ScopedName]):
        for name_ in name.variants():
            if ScopedName(name_, name.scope) in name_set:
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
        Vars(globals={ScopedName(name='open', scope=Scope[], pos=None, cell_no=None)}, locals={ScopedName(name='f', scope=Scope[], pos=None, cell_no=None)})
        """
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.add_to_locals([ScopedName(node.optional_vars.id, None)])

    def visit_Assign(self, node):
        """Assignments - Their targets are locals, their values are globals.

        Example:

        >>> _VarFinder().find_in_source('x = y')
        Vars(globals={ScopedName(name='y', scope=Scope[], pos=None, cell_no=None)}, locals={ScopedName(name='x', scope=Scope[], pos=None, cell_no=None)})
        """
        # collect targets (the 'x' in 'x = a', can be multiple due to 'x = y = a')
        targets = set()
        for target in node.targets:
            # exclude assignment to subscript ('x[1] = a')
            if isinstance(target, ast.Subscript):
                continue
            scope = getattr(target, '_scope', None)
            # exclude assignment to attribute ('x.y = a')
            if isinstance(target, ast.Attribute):
                continue
            targets.update(
                {ScopedName(x.id, scope) for x in Finder(ast.Name).find(target)}
            )
        # find globals in RHS
        sub_globals = {name for name in find_globals(node.value, filter_builtins=False) if not self.in_locals(name)}
        sub_dependencies = set()
        # if a variable on the RHS is one of the targets, increase its counter
        for name in sub_globals:
            if self.in_locals(name):
                continue
            sub_dependencies.add(name)
        self.add_to_locals(targets)
        self.globals.update(sub_dependencies)

    def visit_For(self, node):
        """For loops - Looped over items are locals.

        Example:

        >>> source = '''
        ... for x in range(10):
        ...     ...
        ... '''
        >>> _VarFinder().find_in_source(source)
        Vars(globals={ScopedName(name='range', scope=Scope[], pos=None, cell_no=None)}, locals={ScopedName(name='x', scope=Scope[], pos=None, cell_no=None)})
        """
        self.add_to_locals([ScopedName(x.id, getattr(x, '_scope', None))
                            for x in Finder(ast.Name).find(node.target)])
        for child in node.body:
            self.visit(child)
        self.visit(node.iter)

    def visit_NamedExpr(self, node):
        """The walrus operator - its targets become locals.

        Example:

        >>> import sys
        >>> source = '''
        ... while a := l.pop():
        ...     ...
        ... '''
        >>> sys.version.startswith('3.7') or str(_VarFinder().find_in_source(source))=="Vars(globals={ScopedName(name='l.pop', scope=Scope[], pos=None, cell_no=None)}, locals={ScopedName(name='a', scope=Scope[], pos=None, cell_no=None)})"
        True
        """
        self.add_to_locals([ScopedName(x.id, getattr(x, '_scope', None))
                            for x in Finder(ast.Name).find(node.target)])
        self.visit(node.value)

    def handle_comprehension(self, node):
        """Comprehensions are a special case: Their targets should be ignored. """
        targets = set()
        for gen in node.generators:
            for name in Finder(ast.Name).find(gen.target):
                targets.add(ScopedName(name.id, getattr(name, '_scope', None)))
        sub_globals = set.union(*[find_globals(n, filter_builtins=False)
                                  for n in ast.iter_child_nodes(node)])
        sub_globals = {n for n in sub_globals
                       if not self.in_locals(n)
                       and not self.in_ignoring_attributes(n, targets)}
        self.globals.update(sub_globals)

    def visit_DictComp(self, node):
        """Dict comprehensions.

        Example:

        >>> _VarFinder().find_in_source('{k: v for k, v in foo}')
        Vars(globals={ScopedName(name='foo', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_ListComp(self, node):
        """List comprehensions.

        Example:

        >>> _VarFinder().find_in_source('[x for x in foo]')
        Vars(globals={ScopedName(name='foo', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_SetComp(self, node):
        """Set comprehensions.

        Example:

        >>> _VarFinder().find_in_source('{x for x in foo}')
        Vars(globals={ScopedName(name='foo', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_GeneratorExp(self, node):
        """Generator expressions.

        Example:

        >>> _VarFinder().find_in_source('(x for x in foo)')
        Vars(globals={ScopedName(name='foo', scope=Scope[], pos=None, cell_no=None)}, locals=set())
        """
        self.handle_comprehension(node)

    def visit_Lambda(self, node):
        """Lambda expressions - Their arguments are locals.

        Example:

        >>> vars = _VarFinder().find_in_source('lambda x, y: x + y + foo')
        >>> vars == Vars(globals={ScopedName('foo', None)},
        ...         locals={ScopedName('x', None),
        ...         ScopedName('y', None)})
        True
        """
        scope = getattr(node, '_scope', None)
        posonlyargs = getattr(node.args, 'posonlyargs', [])
        for arg in node.args.args + posonlyargs + node.args.kwonlyargs:
            self.add_to_locals([ScopedName(arg.arg, scope)])
        self.visit(node.body)

    def visit_FunctionDef(self, node):
        """Function definitions. """
        scope = getattr(node, '_scope', None)
        self.add_to_locals([ScopedName(node.name, scope)])
        inner_globals = find_globals(node, filter_builtins=False)
        self.globals.update(v for v in inner_globals if not self.in_locals(v))

    def visit_ClassDef(self, node):
        """Class definitions. """
        scope = getattr(node, '_scope', None)
        self.add_to_locals([ScopedName(node.name, scope)])
        inner_globals = find_globals(node, filter_builtins=False)
        self.globals.update(v for v in inner_globals if not self.in_locals(v))

    def visit_Import(self, node):
        """Import statements.

        Example:

        >>> _VarFinder().find_in_source('import foo')
        Vars(globals=set(), locals={ScopedName(name='foo', scope=Scope[], pos=None, cell_no=None)})
        >>> _VarFinder().find_in_source('import foo as bar')
        Vars(globals=set(), locals={ScopedName(name='bar', scope=Scope[], pos=None, cell_no=None)})
        """
        scope = getattr(node, '_scope', None)
        for name in node.names:
            if name.asname is None:
                self.add_to_locals([ScopedName(name.name, scope)])
            else:
                self.add_to_locals([ScopedName(name.asname, scope)])

    def visit_ImportFrom(self, node):
        """Import-from statements.

        Example:

        >>> _VarFinder().find_in_source('from foo import bar')
        Vars(globals=set(), locals={ScopedName(name='bar', scope=Scope[], pos=None, cell_no=None)})
        >>> _VarFinder().find_in_source('from foo import bar as baz')
        Vars(globals=set(), locals={ScopedName(name='baz', scope=Scope[], pos=None, cell_no=None)})
        """
        scope = getattr(node, '_scope', None)
        for name in node.names:
            if name.asname is None:
                self.add_to_locals([ScopedName(name.name, scope)])
            else:
                self.add_to_locals([ScopedName(name.asname, scope)])


def add_scope_and_pos(variables: List[ScopedName], scoped_name: ScopedName, node):
    """Add missing scope and position information to a list of :class:`ScopedNames` objects.
    """
    result = set()

    for var in variables:
        if var.scope.is_empty():
            var.scope = scoped_name.scope
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                position = ast_utils.get_position(scoped_name.scope.def_source, node)
                var.pos = (position.end_lineno, position.end_col_offset)  # why this?
            else:
                var.pos = scoped_name.pos
        var.cell_no = scoped_name.cell_no
        result.add(var)

    return result


class _Renamer(ast.NodeVisitor):
    # pylint: disable=invalid-name
    """An :class:`ast.NodeTransformer` for renaming things. Used in :func:`rename`.

    :param renaming_function: Function that takes a node and a name and returns a new name.
    :param source: Sourcecode to rename in.
    """

    def __init__(self, renaming_function, source, rename_locals=False, candidates=None):
        self.renaming_function = renaming_function
        self.source = source
        self.res = []
        self.candidates = candidates
        self.rename_locals = rename_locals

    def is_candidate(self, name):
        if self.candidates is None:
            return True
        return name in self.candidates

    def check(self, name, node):
        if not self.is_candidate(name):
            return
        new_name = self.renaming_function(name, node)
        if new_name:
            self.res.append((ast_utils.get_position(self.source, node), new_name))

    def generic_visit(self, node):
        if not self.rename_locals:
            globals_ = {x.name for x in find_globals(node, filter_builtins=False)}
            if self.candidates is None:
                candidates = globals_
            else:
                candidates = self.candidates.intersection(globals_)
        else:
            candidates = self.candidates

        for child in ast.iter_child_nodes(node):
            sub = _Renamer(self.renaming_function, self.source, False, candidates)
            sub.visit(child)
            self.res += sub.res

    def visit_Module(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_Attribute(self, node):
        name = ast_utils.get_source_segment(self.source, node)
        print(name)
        if not self.is_candidate(name.split('.', 1)[0]):
            return

        new_name = self.renaming_function(name, node)
        if new_name:
            self.res.append((ast_utils.get_position(self.source, node), new_name))
            return

        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_Name(self, node):
        """Names.

        Example:

        >>> code = 'x'
        >>> rename(code, from_='x', to='y').strip()
        'y'
        """
        self.check(node.id, node)

    def visit_FunctionDef(self, node):
        """Function definitions.

        Example:

        >>> code = '''
        ... def f(a, b):
        ...     y = p(z)
        ... '''
        >>> rename(code, from_='z', to='u').strip()
        'def f(a, b):\\n    y = p(u)'
        """
        new_name = (
            self.rename_locals
            and self.is_candidate(node.name)
            and self.renaming_function(node.name, node)
        )
        if new_name:
            source_segment = ast_utils.get_source_segment(self.source, node)
            full_position = ast_utils.get_position(self.source, node)
            span = re.search(rf'def ({node.name})\(.*\):.*', source_segment).span(1)
            position = ast_utils.span_to_pos(span, source_segment)
            position.lineno += full_position.lineno - 1
            position.end_lineno += full_position.lineno - 1
            position.col_offset += full_position.col_offset
            if position.lineno == position.end_lineno:
                position.end_col_offset += full_position.col_offset
            self.res.append((position, new_name))

        for arg in node.args.defaults:
            self.visit(arg)
        for sub_node in node.decorator_list:
            self.visit(sub_node)

        globals_ = {x.name for x in find_globals(node, filter_builtins=False)}
        if self.candidates is None:
            candidates = globals_
        else:
            candidates = self.candidates.intersection(globals_)

        for sub_node in node.body:
            sub = _Renamer(self.renaming_function, self.source, False, candidates)
            sub.visit(sub_node)
            self.res += sub.res

    def visit_ClassDef(self, node):
        """Class definitions.

        Example:

        >>> code = '''
        ... class Foo:
        ...     def __init__(self, a, b):
        ...         y = p(z)
        ... '''
        >>> rename(code, from_='z', to='u').strip()
        'class Foo:\\n    def __init__(self, a, b):\\n        y = p(u)'
        """
        new_name = (
            self.rename_locals
            and self.is_candidate(node.name)
            and self.renaming_function(node.name, node)
        )
        if new_name:
            source_segment = ast_utils.get_source_segment(self.source, node)
            full_position = ast_utils.get_position(self.source, node)
            span = re.search(rf'class ({node.name}).*:.*', source_segment).span(1)
            position = ast_utils.span_to_pos(span, source_segment)
            position.lineno += full_position.lineno - 1
            position.end_lineno += full_position.lineno - 1
            position.col_offset += full_position.col_offset
            if position.lineno == position.end_lineno:
                position.end_col_offset += full_position.col_offset
            self.res.append((position, new_name))

        for sub_node in node.bases:
            self.visit(sub_node)
        for sub_node in node.decorator_list:
            self.visit(sub_node)

        for sub_node in node.body:
            globals_ = {x.name for x in find_globals(node, filter_builtins=False)}
            if self.candidates is None:
                candidates = globals_
            else:
                candidates = self.candidates.intersection(globals_)
            sub = _Renamer(self.renaming_function, self.source, False, candidates)
            sub.visit(sub_node)
            self.res += sub.res


def rename(source, tree=None, from_=None, to=None, renaming_function=None, rename_locals=True):
    """Rename things in an AST tree.

    :param source: Source in which to rename things.
    :param tree: The tree in which to rename things (optional, will be parsed from *source* if
        needed).
    :param from_: Rename everything called this ...
    :param to: ... to this (optional, alternatively provide a *renaming_function*, see below).
    :param renaming_function: Function that takes an ast node and a name and returns a new name
        if the name should be changed or *False* else.
    :param rename_locals: If *True*, rename local things, else, only rename globals.
    """
    if tree is None:
        tree = ast.parse(source)

    if renaming_function is None:
        assert from_ is not None and to is not None, 'Must provide either *from_* and *to* or *renaming_function*.'

        def renaming_function(name, _node):
            if name == from_:
                return to
            return False

    renamer = _Renamer(renaming_function, source, rename_locals=rename_locals)
    renamer.visit(tree)
    for pos, name in sorted(renamer.res, key=lambda x: tuple(x[0]), reverse=True):
        source = sourceget.replace(source, name, *pos, one_indexed=True)
    return source


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
    {'one': <...ast.FunctionDef object at 0x...>, 'two': <...ast.FunctionDef object at 0x...>}
    """

    def __init__(self):
        self.methods = {}

    def visit_FunctionDef(self, node):
        self.methods[node.name] = node

    def visit_ClassDef(self, node):
        """Don't visit children of nested classes."""

    def find(self, node: ast.ClassDef) -> Dict['str', ast.FunctionDef]:
        """Do find the methods. """
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        return self.methods


def _filter_builtins(names):
    return {name for name in names if name.toplevel_name not in builtins.__dict__}


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
    >>> find_globals(node) == {ScopedName('a', None), ScopedName('b', None),
    ...                        ScopedName('np.array', None)}
    True

    :param node: AST node to search in.
    :param filter_builtins: If *True*, filter out builtins.
    :return: Set of global ScopedNames.
    """
    if isinstance(node, ast.ClassDef):
        return _find_globals_in_classdef(node)
    globals_ = _VarFinder().find(node).globals
    if filter_builtins:
        globals_ = _filter_builtins(globals_)
    return globals_


def find_codenode(name: ScopedName, full_dump_module_names=None) -> "CodeNode":
    """Find the :class:`CodeNode` corresponding to a :class:`ScopedName` *name*. """
    (source, node), found_name = find_in_scope(name)

    module_name = None
    if full_dump_module_names:
        if isinstance(node, ast.Import):
            module_name = node.names[0].name
        if isinstance(node, ast.ImportFrom):
            module_name = node.module

    if module_name is not None:
        if any(module_name.startswith(mod) for mod in full_dump_module_names):
            return find_codenode(ScopedName(name.name,
                                            Scope.toplevel(module_name)),
                                 full_dump_module_names)
    # find dependencies
    globals_ = find_globals(node)
    globals_ = add_scope_and_pos(globals_, found_name, node)
    return CodeNode(source=source, globals_=globals_, ast_node=node, name=found_name)


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
            raise RuntimeError('Graph has a circle or dangling roots.')  # TODO: improve message
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
    name: ScopedName
    ast_node: Optional[ast.AST] = None

    @classmethod
    def from_source(cls, source, scope, name):
        """Build a `CodeNode` from a source string. """
        scoped_name = ScopedName(name, scope, pos=(0, 0))
        node = ast_utils.ast.parse(source).body[0]
        globals_ = find_globals(node)
        for var in globals_:
            if var.scope.is_empty():
                var.update_scope(scope)

        return cls(
            source=source,
            ast_node=node,
            name=scoped_name,
            globals_=globals_,
        )

    def update_globals(self):
        self.globals_ = find_globals(self.ast_node)
        for var in self.globals_:
            if var.scope is None:
                var.update_scope(self.name.scope)

    def __hash__(self):
        return hash((self.name, tuple(sorted([hash(x) for x in self.globals_]))))

    def __eq__(self, other):
        return (
            (self.name, self.globals_)
            == (other.name, other.globals_)
        )


def _sort(unscoped_graph):
    top = _topsort({k: v.globals_ for k, v in unscoped_graph.items()})

    def sortkey(x):
        val = unscoped_graph[x]
        try:
            return _PRECEDENCE[val.ast_node.__class__](val.name.name, val)
        except KeyError:
            return 'zz' + x.name.lower()

    res = []
    for level in top:
        res += sorted(level, key=sortkey)
    return res


def _deduplicate(keys, graph):
    done = set()
    res = []
    for k in keys:
        if graph[k] in done:
            continue
        done.add(graph[k])
        res.append(k)
    return res


def _remove_empty(keys, graph):
    return [k for k in keys if graph[k].source.strip()]


def _dumps_unscoped(unscoped_graph):
    """Dump an unscoped (see :meth:`CodeGraph.unscope`) :class:`CodeGraph` to a python source
    string. """
    keys = _sort(unscoped_graph)
    keys = _deduplicate(keys, unscoped_graph)
    keys = _remove_empty(keys, unscoped_graph)
    res = ''
    for i, name in enumerate(keys):
        here = unscoped_graph[name]
        res += here.source
        if i < len(keys) - 1:
            next_ = unscoped_graph[keys[i + 1]]
            if isinstance(here.ast_node, (ast.Import, ast.ImportFrom)) \
                    and isinstance(next_.ast_node, (ast.Import, ast.ImportFrom)):
                res += '\n'
            elif isinstance(here.ast_node, ast.Assign) \
                    and isinstance(next_.ast_node, ast.Assign):
                res += '\n'
            else:
                res += '\n\n\n'
    if res:
        return res + '\n'
    return res


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
        name_scope = {(v.name.toplevel_name, v.name.scope) for k, v in self.items()}
        counts = Counter(x[0] for x in name_scope)
        to_rename = set(k for k, c in counts.items() if c > 1)

        for v in self.values():
            assert v.name.pos is not None

        rename_map_k = {
            k: v.name.scope.unscoped(k.name)
            if k.toplevel_name in to_rename else k.name
            for k, v in self.items()
        }

        rename_map_v = {
            v.name: v.name.scope.unscoped(v.name.name)
            if v.name.toplevel_name in to_rename else v.name.name
            for v in self.values()
        }

        ''' WHAT TO DO WITH THIS?
        name_scope_source = defaultdict(dict)
        counts = Counter()
        for k, v in self.items():
            len_ = name_scope_source[v.name.name, v.name.scope].get(
                v.source,
                len(name_scope_source[v.name.name, v.name.scope])
            )
            name_scope_source[v.name.name, v.name.scope][v.source] = len_
            if len_ != 0:
                rename_map[k] = f'{rename_map[k]}_{len_}'
                to_rename.add(v.name.name)
        '''

        def unscope(name, scope):
            if name in to_rename:
                return scope.unscoped(name)
            return name

        def renaming_function(name, node, k, k_unscoped):
            if hasattr(node, '_scope') and node._scope != k.scope:
                return False
            if name != k.name:
                return False
            if name == k_unscoped:
                return False
            return k_unscoped

        res = {}
        for k, v in self.items():
            v_unscoped = rename_map_v[v.name]
            parsed = ast.parse(v.source).body[0]
            copied = copy.deepcopy(v.ast_node)
            for a, b in zip(ast.walk(parsed), ast.walk(copied)):
                ast.copy_location(b, a)
                assert type(a) == type(b)

            def rn(name, node):
                new_name = renaming_function(name, node, v.name, v_unscoped)
                if new_name:
                    return new_name
                for n in list(v.globals_):
                    new_name = renaming_function(name, node, n, rename_map_k[n])
                    if new_name:
                        return new_name
                return False

            code = rename(v.source, copied, renaming_function=rn, rename_locals=True)
            res[k] = CodeNode(code, v.globals_, ast_node=v.ast_node,
                              name=ScopedName(v_unscoped, Scope.empty()))
        return res

    def dumps(self):
        """Create a python source string with the contents of the graph. """
        return _dumps_unscoped(self._unscoped())

    def real_names(self):
        return {v.name: v for v in self.values()}

    def from_source(self, target: str, scope: Optional[Scope] = None, name='__out'):
        if scope is None:
            scope = Scope.toplevel('__main__')
        start = CodeNode.from_source(target, scope, name)
        graph = self.build(list(start.globals_))
        graph[ScopedName(name, scope=scope)] = start
        return graph

    def build(self, target: Union[List[ScopedName], ScopedName]):
        """Build a codegraph corresponding to a :class:`ScopedName` or a list of
        :class:`ScopedName`s.

        The name(s) will be searched for in its (their) scope.
        """
        done = set()

        if isinstance(target, list):
            todo = set(target)
        else:
            todo = {target}

        while todo:
            # we know this already - go on
            next_name = todo.pop()

            if next_name in done:
                continue

            # find how next_var came into being
            next_codenode = find_codenode(next_name)
            self[next_name] = next_codenode

            todo.update(next_codenode.globals_)
            done.add(next_name)

        return self

    def print(self):
        """Print the graph (for debugging). """
        for k, v in self.items():
            print(f'{k}:')
            print()
            print(v.name)
            print()
            print(v.source)
            print()
            for dep in v.globals_:
                print(f'  {dep}')
            print()
            print('--------')
            print()
