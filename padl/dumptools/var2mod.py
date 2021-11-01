import ast
import builtins
from collections import Counter, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
import sys
from typing import Optional

from padl.dumptools import symfinder

try:
    unparse = ast.unparse
except AttributeError:  # python < 3.9
    from astunparse import unparse


class Finder(ast.NodeVisitor):
    """Class for finding ast nodes of a given type in an ast tree.

    Example:

    >>> f = Finder(ast.Name).find(ast.parse('x(y)'))
    [<_ast.Name at ...>, <_ast.Name at ...>]
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
            self.locals.add(arg.arg)
        if node.args.vararg is not None:
            self.locals.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self.locals.add(node.args.kwarg.arg)
        for n in ast.iter_child_nodes(node):
            self.visit(n)
        return Vars(self.globals, self.locals)

    def visit_Name(self, node):
        if node.id not in self.locals:
            self.globals.add(node.id)

    def visit_withitem(self, node):
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.locals.add(node.optional_vars.id)

    def visit_Assign(self, node):
        for target in node.targets:
            self.locals.update([x.id for x in Finder(ast.Name).find(target)])
        self.visit(node.value)

    def visit_For(self, node):
        self.locals.update([x.id for x in Finder(ast.Name).find(node.target)])
        for child in node.body:
            self.visit(child)

    def visit_NamedExpr(self, node):
        self.locals.update(Finder(ast.Name).find(node.target))

    def visit_comprehension(self, node):
        """Special case for comprehension - comprehension targets should be ignored. """
        targets = set()
        for gen in node.generators:
            for name in Finder(ast.Name).find(gen.target):
                targets.add(name.id)
        all_ = set(x.id for x in Finder(ast.Name).find(node))
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
            self.locals.add(arg.arg)
        self.visit(node.body)

    def visit_FunctionDef(self, node):
        self.locals.add(node.name)
        inner_globals = find_globals(node)
        self.globals.update(inner_globals - self.locals)

    def visit_ClassDef(self, node):
        self.locals.add(node.name)
        inner_globals = find_globals(node)
        self.globals.update(inner_globals - self.locals)

    def visit_Import(self, node):
        for name in node.names:
            if name.asname is None:
                self.locals.add(name.name)
            else:
                self.locals.add(name.asname)

    def visit_ImportFrom(self, node):
        for name in node.names:
            if name.asname is None:
                self.locals.add(name.name)
            else:
                self.locals.add(name.asname)


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
        globals_ = {x for x in globals_ if x not in builtins.__dict__}
    return globals_


def find_globals(node: ast.AST, filter_builtins=True):
    """Find all globals used below a node. """
    if isinstance(node, ast.ClassDef):
        return _find_globals_in_classdef(node)
    globals_ = _VarFinder().find(node).globals
    if filter_builtins:
        globals_ = {x for x in globals_ if x not in builtins.__dict__}
    return globals_


def build_codegraph(x: set, module=None, all_=None, update_callback=None):
    if module is None:
        module = sys.modules['__main__']
    if all_ is None:
        all_ = {}
    todo = set(x)
    while todo and (next_ := todo.pop()):
        if next_ in all_:
            continue
        if update_callback is not None and update_callback(next_, module):
            pass
        else:
            source, node = symfinder.find(next_, module)
            globals_ = find_globals(node)
            all_[next_] = (source, globals_, node)
        todo.update(globals_)
    return all_


def _get_nodes_without_in_edges(graph):
    """Get all nodes in directed graph *graph* that don't have incoming edges.

    The graph is represented by a dict mapping nodes to incoming edges.
    """
    nextlevel = set()
    for node, deps in graph.items():
        if not deps:
            nextlevel.add(node)
    filtered_graph = {}
    for node, deps in graph.items():
        if node in nextlevel:
            continue
        filtered_graph[node] = \
            {dep for dep in deps if dep not in nextlevel}
    return nextlevel, filtered_graph


def _topsort(graph):
    """Topologically sort a graph represented by a dict mapping nodes to incoming edges.
    Careful: Doesn't check for loops.
    """
    levels = []
    while graph:
        nextlevel, graph = _get_nodes_without_in_edges(graph)
        levels.append(nextlevel)
    return levels


# sort precedence
_PRECEDENCE = {
    ast.Import: lambda k, v: '1' + v[2].names[0].name.lower() + k.lower(),
    ast.ImportFrom: lambda k, v: '1' + v[2].module.lower() + k.lower(),
    ast.Assign: lambda k, _v: '2' + k.lower()
}


def _sort(graph):
    top = _topsort({k: v[1] for k, v in graph.items()})

    def sortkey(x):
        val = graph[x]
        try:
            return _PRECEDENCE[val[2].__class__](x, val)
        except KeyError:
            return 'zz' + x.lower()

    res = []
    for level in top:
        res += sorted(level, key=sortkey)
    return res


def unscope_graph(graph, scopemap):
    """Create a version of *graph* where all non-top level variables are renamed (by prepending
    the scope) to prevent conflicts."""
    counts = Counter(x[0] for x in graph)
    to_rename = set(k for k, c in counts.items() if c > 1)
    scopemap = {**scopemap}
    scopemap.update({(k[0], v): v for k, v in scopemap})
    def unscope(name, scope):
        if name in to_rename:
            return scope.unscoped(name)
        return name
    res = {}
    for k, v in graph.items():
        changed = False
        k_unscoped = unscope(*k)
        changed = changed or k_unscoped != k[0]
        code = v.source
        tree = ast.parse(code)
        rename(tree, k[0], k_unscoped, rename_locals=True)
        vars_ = set()
        for var in list(v.globals_):
            var_unscoped = unscope(var[0], scopemap[var])
            changed = changed or var_unscoped != var[0]
            rename(tree, var[0], var_unscoped)
            vars_.add(var_unscoped)
        if changed:
            code = unparse(tree).strip('\n')
        res[k_unscoped] = (code, vars_, v.ast_node)
    return res


@dataclass
class CodeNode:
    source: str
    globals_: set
    ast_node: Optional[ast.AST] = None

    @classmethod
    def from_source(cls, source, scope):
        node = ast.parse(source).body[0]
        globals_ = {
            (var, scope)
            for var in find_globals(node)
        }

        return cls(
            source=source,
            ast_node=node,
            globals_=globals_
        )


def dumps_graph(graph):
    sorted_ = _sort(graph)
    res = ''
    for i, name in enumerate(sorted_):
        here = graph[name]
        res += here[0]
        if i < len(sorted_) - 1:
            next_ = graph[sorted_[i + 1]]
            if isinstance(here[2], (ast.Import, ast.ImportFrom)) \
                    and isinstance(next_[2], (ast.Import, ast.ImportFrom)):
                res += '\n'
            elif isinstance(here[2], ast.Assign) \
                    and isinstance(next_[2], ast.Assign):
                res += '\n'
            else:
                res += '\n\n\n'
    return res + '\n'


def dumps(x: str, module=None):
    graph = build_codegraph({x}, module)
    return dumps_graph(graph)
