import ast
import builtins
from collections import Counter, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, List, Tuple

from padl.dumptools.symfinder import find_in_scope, ScopedName, Scope

try:
    unparse = ast.unparse
except AttributeError:  # python < 3.9
    from astunparse import unparse


class Finder(ast.NodeVisitor):
    """Class for finding ast nodes of a given type in an ast tree.

    Example:

    >>> f = Finder(ast.Name).find(ast.parse('x(y)'))
    >>> all([isinstance(x, ast.Name) for x in f])
    True
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
        """Find subnodes of node. """
        self.visit(node)
        return self.result

    def get_source_segments(self, source):
        nodes = self.find(ast.parse(source))
        return [
            (ast.get_source_segment(source, node),
             (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
            )
            for node in nodes
        ]


Vars = namedtuple('Vars', 'globals locals')


class _VarFinder(ast.NodeVisitor):
    """NodeVisitor that traverses all nodes subnodes and finds all named things. """

    def __init__(self):
        super().__init__()
        self.globals = set()
        self.locals = set()

    def find(self, node):
        """Find all globals and locals.

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

    def _find_in_function_def(self, node):
        """Special case: exclude args from globals. """
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self.locals.add((arg.arg, 0))
        if node.args.vararg is not None:
            self.locals.add((node.args.vararg.arg, 0))
        if node.args.kwarg is not None:
            self.locals.add((node.args.kwarg.arg, 0))
        for n in ast.iter_child_nodes(node):
            self.visit(n)
        return Vars(self.globals, self.locals)

    def visit_Name(self, node):
        if (node.id, 0) not in self.locals:
            self.globals.add((node.id, 0))

    def visit_withitem(self, node):
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.locals.add((node.optional_vars.id, 0))

    def visit_Assign(self, node):
        # collect targets (the 'x' in 'x = a', can be multiple due to 'x = y = a')
        targets = set()
        for target in node.targets:
            # exclude assignment to subscript ('x[1] = a')
            if isinstance(target, ast.Subscript):
                continue
            # exclude assignment to attribute ('x.y = a')
            if isinstance(target, ast.Attribute):
                continue
            targets.update(
                {(x.id, 0) for x in Finder(ast.Name).find(target)}
            )
        # find globals in RHS
        sub_globals = find_globals(node.value) - self.locals
        sub_dependencies = set()
        # if a variable on the RHS is one of the targets, increase its counter
        for name, i in sub_globals:
            if (name, i) in targets:
                sub_dependencies.add((name, i + 1))
            else:
                sub_dependencies.add((name, i))
        self.locals.update(targets)
        self.globals.update(sub_dependencies - self.locals)

    def visit_For(self, node):
        self.locals.update([(x.id, 0) for x in Finder(ast.Name).find(node.target)])
        for child in node.body:
            self.visit(child)

    def visit_NamedExpr(self, node):
        self.locals.update([(x.id, 0) for x in Finder(ast.Name).find(node.target)])

    def visit_comprehension(self, node):
        """Special case for comprehension - comprehension targets should be ignored. """
        targets = set()
        for gen in node.generators:
            for name in Finder(ast.Name).find(gen.target):
                targets.add((name.id, 0))
        all_ = set((x.id, 0) for x in Finder(ast.Name).find(node))
        self.globals.update(all_ - targets - self.locals)

    def visit_DictComp(self, node):
        self.visit_comprehension(node)

    def visit_ListComp(self, node):
        self.visit_comprehension(node)

    def visit_SetComp(self, node):
        self.visit_comprehension(node)

    def visit_GeneratorExp(self, node):
        pass

    def visit_Lambda(self, node):
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self.locals.add((arg.arg, 0))
        self.visit(node.body)

    def visit_FunctionDef(self, node):
        self.locals.add((node.name, 0))
        inner_globals = find_globals(node)
        self.globals.update(inner_globals - self.locals)

    def visit_ClassDef(self, node):
        self.locals.add((node.name, 0))
        inner_globals = find_globals(node)
        self.globals.update(inner_globals - self.locals)

    def visit_Import(self, node):
        for name in node.names:
            if name.asname is None:
                self.locals.add((name.name, 0))
            else:
                self.locals.add((name.asname, 0))

    def visit_ImportFrom(self, node):
        for name in node.names:
            if name.asname is None:
                self.locals.add((name.name, 0))
            else:
                self.locals.add((name.asname, 0))


class _Renamer(ast.NodeTransformer):
    def __init__(self, from_, to, rename_locals):
        self.from_ = from_
        self.to = to
        self.rename_locals = rename_locals

    def visit_arg(self, node):
        if node.arg == self.from_:
            return ast.arg(**{**node.__dict__, 'arg': self.to})
        return node

    def visit_Name(self, node):
        if node.id == self.from_:
            return ast.Name(**{**node.__dict__, 'id': self.to})
        return node

    def visit_FunctionDef(self, node):
        if self.rename_locals and node.name == self.from_:
            name = self.to
        else:
            name = node.name
        if not self.rename_locals:
            globals_ = find_globals(node)
            if self.from_ not in globals_:
                return node
        return ast.FunctionDef(**{**node.__dict__,
                                  'body': rename(node.body, self.from_, self.to),
                                  'name': name})

    def visit_ClassDef(self, node):
        if self.rename_locals and node.name == self.from_:
            name = self.to
        else:
            name = node.name
        if not self.rename_locals:
            globals_ = find_globals(node)
            if self.from_ not in globals_:
                return node
        return ast.ClassDef(**{**node.__dict__,
                               'body': rename(node.body, self.from_, self.to),
                               'name': name})


def rename(tree, from_, to, rename_locals=False):
    if not rename_locals:
        globals_ = find_globals(tree)
        if from_ not in globals_:
            return tree
    if isinstance(tree, Iterable) and not isinstance(tree, str):
        return [rename(x, from_, to) for x in tree]
    elif not isinstance(tree, ast.AST):
        return tree
    renamer = _Renamer(from_, to, rename_locals)
    return renamer.visit(tree)


class _MethodFinder(ast.NodeVisitor):
    """Find all methods in a class node. """

    def __init__(self):
        self.methods = {}

    def visit_FunctionDef(self, node):
        self.methods[node.name] = node

    def find(self, node: ast.ClassDef):
        self.visit(node)
        return self.methods


def _find_globals_in_classdef(node: ast.ClassDef, filter_builtins=True):
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
        globals_ = {(x, i) for x, i in globals_ if x not in builtins.__dict__}
    return globals_


def find_globals(node: ast.AST, filter_builtins=True):
    """Find all globals used below a node. """
    if isinstance(node, ast.ClassDef):
        return _find_globals_in_classdef(node)
    globals_ = _VarFinder().find(node).globals
    if filter_builtins:
        globals_ = {(x, i) for x, i in globals_ if x not in builtins.__dict__}
    return globals_


def increment_same_name_var(variables: List[Tuple[str, int]], scoped_name: ScopedName):
    """Go through *variables* and increment the the counter for those with the same name as
    *scoped_name* by *scoped_name.n*.

    Example:

    >>> import padl as somemodule
    >>> out = increment_same_name_var({('a', 1), ('b', 2)}, ScopedName('b', somemodule, 2))
    >>> isinstance(out, set)
    True
    >>> {(x.name, x.n) for x in out} == {('a', 1), ('b', 4)}
    True
    """
    result = set()
    for var, n in variables:
        if var == scoped_name.name:
            result.add(ScopedName(var, scoped_name.scope, n + scoped_name.n))
        else:
            result.add(ScopedName(var, scoped_name.scope, n))
    return result


def find_codenode(name: ScopedName):
    (source, node), scope_of_next_var = find_in_scope(name)

    # find dependencies
    globals_ = find_globals(node)
    next_name = ScopedName(name.name, scope_of_next_var, name.n)
    globals_ = increment_same_name_var(globals_, next_name)

    return CodeNode(source=source, globals_=globals_, ast_node=node, scope=scope_of_next_var)


def _get_nodes_without_in_edges(graph):
    """Get all nodes in directed graph *graph* that don't have incoming edges.

    The graph is represented by a dict mapping nodes to incoming edges.
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


def _topsort(graph):
    """Topologically sort a graph represented by a dict mapping nodes to incoming edges.
    """
    levels = []
    graphlen = len(graph)
    while graph:
        nextlevel, graph = _get_nodes_without_in_edges(graph)
        if graphlen == len(graph):  # graph didn't shrink
            raise RuntimeError('Graph has a circle.')
        graphlen = len(graph)
        levels.append(nextlevel)
    return levels


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


@dataclass
class CodeNode:
    source: str
    globals_: set
    scope: Optional[Scope] = None
    ast_node: Optional[ast.AST] = None

    @classmethod
    def from_source(cls, source, scope):
        node = ast.parse(source).body[0]
        globals_ = {
            ScopedName(var, scope, n)
            for var, n in find_globals(node)
        }

        return cls(
            source=source,
            ast_node=node,
            globals_=globals_,
            scope=scope
        )


def _dumps_unscoped(unscoped_graph):
    """Dump an unscoped (see :meth:`CodeGraph.unscope`) codegraph to a python source string. """
    sorted_ = _sort(unscoped_graph)
    res = ''
    for i, name in enumerate(sorted_):
        here = unscoped_graph[name]
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
            res[k_unscoped, k.n] = CodeNode(code, vars_, ast_node=v.ast_node)
        return res

    def dumps(self):
        """Create a python source string with the contents of the graph. """
        return _dumps_unscoped(self._unscoped())

    @classmethod
    def build(cls, scoped_name: ScopedName):
        """Build a codegraph corresponding to a name.

        The name will be searched for in its scope.
        """
        graph = cls()
        done = set()

        todo = {scoped_name}

        while todo and (next_name := todo.pop()):
            # we know this already - go on
            if next_name in done:
                continue

            # find how next_var came into being
            next_codenode = find_codenode(next_name)
            graph[next_name] = next_codenode

            todo.update(next_codenode.globals_)
            done.add(next_name)

        return graph

    def print(self):
        for k, v in self.items():
            print(f'{k.name} {k.scope} {k.n}:')
            print()
            print(v.source)
            print()
            for d in v.globals_:
                print(f'{d.name} {d.scope} {d.n}:')
            print()
            print('--------')
            print()
