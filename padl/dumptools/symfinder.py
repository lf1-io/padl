# pylint: disable=invalid-name
"""Module for symbolically finding python entities in python source code given their name.

A thing in python can get its name in various ways:

 - it's defined as a function
 - it's defined as a class
 - it's assigned
 - it is imported
 - it's created in a with statement
 - it's created in a for loop

This module defines subclasses of the `_ThingFinder` class, which allow to identify these cases
in an AST tree of a source code.

Finding names in code then corresponds to building the AST tree of the code and using the
`_ThingFinder` subclasses to identify if and how the names were created.

The main function to use is `find`, which will find a name in a module or the current ipython
history.
"""

import ast
from dataclasses import dataclass
from math import inf
import sys
from textwrap import dedent
from types import ModuleType
from typing import List, Tuple

from padl.dumptools import ast_utils, sourceget


class _ThingFinder(ast.NodeVisitor):
    """Class for finding python "things" in a given source code given a variable name.

    :param source: The source in which the thing is searched.
    :param var_name: The name of the variable that's searched for.
    :param max_n: Stop searching after this number of statements from the bottom
        of the module.
    """

    def __init__(self, source: str, var_name: str, max_n: int = inf):
        self.source = source
        self.var_name = var_name
        self.statement_n = 0
        self.max_n = max_n
        self._result = None

    def found_something(self) -> bool:
        """*True* if something was found. """
        return self._result is not None

    def visit_Module(self, node):
        """Visit each statement in a module's body, from top to bottom and stop at "max_n". """
        for i, statement in enumerate(node.body[::-1]):
            if i > self.max_n:
                return
            self.visit(statement)
            if self.found_something():
                self.statement_n = i
                return

    def visit_With(self, node):
        """With is currently not supported - raises an error if the "thing" is defined in the
        head of a "with"-statement. """
        for item in node.items:
            if item.optional_vars is not None and item.optional_vars.id == self.var_name:
                raise NotImplementedError(f'"{self.var_name}" is defined in the head of a with'
                                          ' statement. This is currently not supported.')
        for subnode in node.body:
            self.visit(subnode)

    def visit_ClassDef(self, node):
        """Don't search class definitions (as that's another scope). """

    def visit_FunctionDef(self, node):
        """Don't search function definitions (as that's another scope). """

    def deparse(self) -> str:
        """Get the source snipped corresponding to the found node. """
        return ast_utils.get_source_segment(self.source, self._result)

    def node(self) -> ast.AST:
        """Get the found node. """
        # TODO: correctly inherit (is not always _result)
        return self._result


class _NameFinder(_ThingFinder):
    pass


class _FunctionDefFinder(_NameFinder):
    """Class for finding a *function definition* of a specified name in an AST tree.

    Example:

    >>> source = '''
    ... import baz
    ...
    ... def foo(x):
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = 100'''
    >>> finder = _FunctionDefFinder(source, 'foo')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'def foo(x):\\n    ...'
    >>> finder.node()  # doctest: +ELLIPSIS
    <_ast.FunctionDef object at 0x...>
    """

    def visit_FunctionDef(self, node):
        if node.name == self.var_name:
            self._result = node

    def deparse(self):
        res = ''
        res = ast_utils.get_source_segment(self.source, self._result)
        # for py 3.8+, the decorators are not included, we need to add them
        if not res.lstrip().startswith('@'):
            for decorator in self._result.decorator_list[::-1]:
                res = f'@{ast_utils.get_source_segment(self.source, decorator)}\n' + res
        return _fix_indent(res)


def _fix_indent(source):
    """Fix the indentation of functions that are wrongly indented.

    This can happen with :func:`ast_utils.get_source_segment`.
    """
    lines = source.lstrip().split('\n')
    res = []
    for line in lines:
        if line.startswith('@'):
            res.append(line.lstrip())
            continue
        if line.lstrip().startswith('def ') or line.lstrip().startswith('class '):
            res.append(line.lstrip())
        break
    lines = lines[len(res):]
    rest_dedented = dedent('\n'.join(lines))
    res = res + ['    ' + line for line in rest_dedented.split('\n')]
    return '\n'.join(line.rstrip() for line in res)


class _ClassDefFinder(_NameFinder):
    """Class for finding a *class definition* of a specified name in an AST tree.

    Example:

    >>> source = '''
    ... import baz
    ...
    ... class Foo:
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = 100'''
    >>> finder = _ClassDefFinder(source, 'Foo')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'class Foo:\\n    ...'
    >>> finder.node()  # doctest: +ELLIPSIS
    <_ast.ClassDef object at 0x...>
    """

    def visit_ClassDef(self, node):
        if node.name == self.var_name:
            self._result = node

    def deparse(self):
        res = ''
        res = ast_utils.get_source_segment(self.source, self._result)
        res = _fix_indent(res)
        # for py 3.8+, the decorators are not included, we need to add them
        if not res.lstrip().startswith('@'):
            for decorator in self._result.decorator_list[::-1]:
                res = f'@{ast_utils.get_source_segment(self.source, decorator)}\n' + res
        return res


class _ImportFinder(_NameFinder):
    """Class for finding a *module import* of a specified name in an AST tree.

    Works with normal imports ("import x") and aliased imports ("import x as y").

    Example:

    >>> source = '''
    ... import baz as boo
    ...
    ... class Foo:
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = 100'''
    >>> finder = _ImportFinder(source, 'boo')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'import baz as boo'
    """

    def visit_Import(self, node):
        for name in node.names:
            if name.asname == self.var_name:
                self._result = name
                return
            if name.asname is None and name.name == self.var_name:
                self._result = name
                return

    def deparse(self):
        name = self._result
        res = f'import {name.name}'
        if name.asname is not None:
            res += f' as {name.asname}'
        return res

    def node(self):
        # TODO: cache deparse?
        node = ast.parse(self.deparse()).body[0]
        if node.names[0].asname is None:
            node._globalscope = True
        return node


class _ImportFromFinder(_NameFinder):
    """Class for finding a *from import* of a specified name in an AST tree.

    Example:

    >>> source = '''
    ... from boo import baz as hoo, bup
    ...
    ... class Foo:
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = 100'''
    >>> finder = _ImportFromFinder(source, 'hoo')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'from boo import baz as hoo'
    """

    def visit_ImportFrom(self, node):
        for name in node.names:
            if name.asname == self.var_name:
                self._result = (node.module, name)
                return
            if name.asname is None and name.name == self.var_name:
                self._result = (node.module, name)
                return

    def deparse(self):
        module, name = self._result
        res = f'from {module} import {name.name}'
        if name.asname is not None:
            res += f' as {name.asname}'
        return res

    def node(self):
        # TODO: cache deparse?
        node = ast.parse(self.deparse()).body[0]
        # the scope does not matter here
        if node.names[0].asname is None:
            node._globalscope = True
        return node


class _AssignFinder(_NameFinder):
    """Class for finding a *variable assignment* of a specified name in an AST tree.

    Example:

    >>> source = '''
    ... import baz
    ...
    ... class Foo:
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = 100'''
    >>> finder = _AssignFinder(source, 'X')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'X = 100'
    >>> finder.node()  # doctest: +ELLIPSIS
    <_ast.Assign object at 0x...>
    """

    def visit_Assign(self, node):
        for target in node.targets:
            if self._parse_target(target):
                self._result = node
                return

    def visit_AnnAssign(self, node):
        if self._parse_target(node.target):
            self._result = node
            return

    def _parse_target(self, target):
        if isinstance(target, ast.Name) and target.id == self.var_name:
            return True
        if isinstance(target, ast.Tuple):
            for sub_target in target.elts:
                if self._parse_target(sub_target):
                    return True
        return False

    def deparse(self):
        try:
            source = self._result._source
        except AttributeError:
            source = self.source
        return ast_utils.get_source_segment(source, self._result)


class _SetAttribute(ast.NodeVisitor):
    """Class for setting an attribute on all nodes in an ast tree.

    This is being used in :meth:`Scope.from_source` to tag nodes with the scope they were found in.

    Example:

    >>> tree = ast.parse('a = f(0)')
    >>> _SetAttribute('myattribute', True).visit(tree)
    >>> tree.body[0].targets[0].myattribute
    True
    >>> tree.body[0].value.myattribute
    True
    """

    def __init__(self, attr, value):
        self.attr = attr
        self.value = value

    def generic_visit(self, node):
        setattr(node, self.attr, self.value)
        super().generic_visit(node)


class _CallFinder(_ThingFinder):
    """Class for finding a *call* in an AST tree.

    Example:

    >>> source = '''
    ... import baz
    ...
    ... class Foo:
    ...     ...
    ...
    ... def bar(y):
    ...     ...
    ...
    ... X = baz(100)'''
    >>> finder = _CallFinder(source, 'baz')
    >>> node = ast.parse(source)
    >>> finder.visit(node)
    >>> finder.found_something()
    True
    >>> finder.deparse()
    'baz(100)'
    >>> finder.node()  # doctest: +ELLIPSIS
    <_ast.Call object at 0x...>
    """

    def visit_Call(self, node):
        if node.func.id == self.var_name:
            self._result = node

    def find(self):
        tree = ast.parse(self.source)
        self.visit(tree)
        if self.found_something():
            return self._get_name(self._result), (*_get_call_signature(self.source),)
        raise NameNotFound(f'Did not find call of "{self.var_name}".')

    def _get_name(self, call: ast.Call):
        return ast_utils.get_source_segment(self.source, call.func)


def _get_call_assignments(args, source, values, keywords):
    argnames = [x.arg for x in args.args]
    try:
        pos_only_argnames = [x.arg for x in args.posonlyargs]
    except AttributeError:
        pos_only_argnames = []
    all_argnames = pos_only_argnames + argnames
    defaults = {
        name: ast_utils.get_source_segment(source, val)
        for name, val in zip(argnames[::-1], args.defaults[::-1])
    }
    kwonly_defaults = {
        ast_utils.get_source_segment(source, name): ast_utils.get_source_segment(source, val)
        for name, val in zip(args.kwonlyargs, args.kw_defaults)
        if val is not None
    }
    res = {}
    for name, val in kwonly_defaults.items():
        try:
            res[name] = keywords[name]
        except KeyError:
            res[name] = val

    for name, val in zip(all_argnames, values):
        res[name] = val

    if args.vararg is not None:
        res[args.vararg.arg] = '[' + ', '.join(values[len(all_argnames):]) + ']'

    kwargs = {}
    for name, val in keywords.items():
        if name in res:
            continue
        if name in argnames:
            res[name] = val
        else:
            kwargs[name] = val

    if kwargs and not set(kwargs) == {None}:
        assert args.kwarg is not None, 'Keyword args given, but no **kwarg present.'
        res[args.kwarg.arg] = '{' + ', '.join(f"'{k}': {v}" for k, v in kwargs.items()) + '}'

    for name, val in defaults.items():
        if name not in res:
            res[name] = val

    return res


def _get_call_signature(source: str):
    """Get the call signature of a string containing a call.

    :param source: String containing a call (e.g. "a(2, b, 'f', c=100)")
    :returns: A tuple with a list of positional arguments and a list of keyword arguments.

    Example:

    >>> _get_call_signature("a(2, b, 'f', c=100)")
    (['2', 'b', "'f'"], {'c': '100'})
    """
    call = ast.parse(source).body[0].value
    if not isinstance(call, ast.Call):
        return [], {}
    args = [ast_utils.get_source_segment(source, arg) for arg in call.args]
    kwargs = {
        kw.arg: ast_utils.get_source_segment(source, kw.value) for kw in call.keywords
    }
    return args, kwargs


class Scope:
    """A scope.

    Scope objects can be used to find names that are not defined globally in a module, but
    nested, for example within a function body.

    It contains the module, the source string and a "scopelist".
    """

    def __init__(self, module: ModuleType, def_source: str,
                 scopelist: List[Tuple[str, ast.AST]]):
        self.module = module
        self.def_source = def_source
        self.scopelist = scopelist

    @classmethod
    def toplevel(cls, module):
        """Create a top-level scope (i.e. module level, no nesting). """
        if isinstance(module, str):
            module = sys.modules[module]
        return cls(module, '', [])

    @classmethod
    def empty(cls):
        """Create the empty scope (a scope with no module and no nesting). """
        return cls(None, '', [])

    @classmethod
    def from_source(cls, def_source, lineno, call_source, module=None, drop_n=0,
                    calling_scope=None):
        """Create a `Scope` object from source code.

        :param def_source: The source string containing the scope.
        :param lineno: The line number to get the scope from.
        :param call_source: The source of the call used for accessing the scope.
        :param module: The module.
        :param drop_n: Number of levels to drop from the scope.
        """
        tree = ast.parse(def_source)
        branch = _find_branch(tree, lineno, def_source)
        function_defs = [x for x in branch if isinstance(x, ast.FunctionDef)]
        if drop_n > 0:
            function_defs = function_defs[:-drop_n]

        if not function_defs:
            return cls.toplevel(module)

        # get call assignments for inner function
        # def f(a, b, c=3):
        #    ...
        # and
        # -> f(1, 2)
        # makes
        # a = 1
        # b = 2
        # c = 3
        # ...
        values, keywords = _get_call_signature(call_source)
        args = function_defs[-1].args
        assignments = _get_call_assignments(args, def_source, values, keywords)
        call_assignments = []
        for k, v in assignments.items():
            src = f'{k} = {v}'
            assignment = ast.parse(src).body[0]
            assignment._source = src
            _SetAttribute('_scope', calling_scope).visit(assignment.value)
            call_assignments.append(assignment)

        scopelist = []
        for fdef in function_defs[::-1]:
            module_node = ast.Module()
            module_node.body = []
            module_node.body = fdef.body
            scopelist.append((fdef.name, module_node))

        # add call assignments to inner scope
        scopelist[0][1].body = call_assignments + scopelist[0][1].body

        return cls(module, def_source, scopelist)

    def from_level(self, i: int) -> 'Scope':
        """Return a new scope starting at level *i* of the scope hierarchy. """
        return type(self)(self.module, self.def_source, self.scopelist[i:])

    def up(self) -> 'Scope':
        """Return a new scope one level up in the scope hierarchy. """
        return type(self)(self.module, self.def_source, self.scopelist[1:])

    def global_(self) -> 'Scope':
        """Return the global scope surrounding *self*. """
        return type(self)(self.module, self.def_source, [])

    def is_global(self) -> bool:
        """*True* iff the scope is global. """
        return len(self.scopelist) == 0

    @property
    def module_name(self) -> str:
        """The name of the scope's module. """
        if self.module is None:
            return ''
        return getattr(self.module, '__name__', '__main__')

    def unscoped(self, varname: str) -> str:
        """Convert a variable name in an "unscoped" version by adding strings representing
        the containing scope. """
        if not self.scopelist and self.module_name in ('', '__main__'):
            return varname
        return f'{"_".join(x[0] for x in [(self.module_name.replace(".", "_"), 0)] + self.scopelist)}_{varname}'

    def __repr__(self):
        return f'Scope[{".".join(x[0] for x in [(self.module_name, 0)] + self.scopelist[::-1])}]'

    def __len__(self):
        return len(self.scopelist)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


@dataclass
class ScopedName:
    """A name with a scope and a counter. The "name" is the name of the item, the scope is its
    :class:`Scope` and the counter counts the items with the same name, in the same scope,
    from most recent on up.

    Example - the following::

        a = 1

        def f(x):
            a = 2

        a = a + 1

    contains four scoped names:

        - The "a" of `a = a + 1`, with `name = a`, module-level scope and `n = 0` (it is the most
          recent "a" in the module level scope).
        - The "a" in the function body, with `name = a`, function f scope and `n = 0` (it is the
          most recent "a" in "f" scope).
        - the function name "f", module level scope, `n = 0`
        - the "a" of `a = 1`, with `name = a`, module-level scope and `n = 1` (as it's the second
          most recent "a" in its scope).
    """
    name: str
    scope: Scope
    n: int = 0

    def __hash__(self):
        return hash((self.name, self.scope, self.n))

    def __eq__(self, other):
        return (self.name, self.scope, self.n) == (other.name, other.scope, other.n)


def find_in_scope(name: ScopedName):
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    scope *scope*.

    :param scope: Name (with scope) of the variable to look for.
    """
    scope = name.scope
    i = name.n
    for _scopename, tree in scope.scopelist:
        try:
            res = find_in_source(name.name, name.scope.def_source, tree=tree, i=i,
                                 return_partial=True)
            if isinstance(res, int):
                i = res
                continue
            source, node, name = res
            if getattr(node, '_globalscope', False):
                scope = Scope.empty()

            return (source, node), scope, name
        except NameNotFound:
            scope = scope.up()
            continue
    if scope.module is None:
        raise NameNotFound(f'{name.name} not found in function hierarchy.')
    source, node, name = find(name.name, scope.module, i)
    if getattr(node, '_globalscope', False):
        scope = Scope.empty()
    else:
        scope = getattr(node, '_scope', scope.global_())
    return (source, node), scope, name


def replace_star_imports(tree: ast.Module):
    """Replace star imports in the tree with their written out forms.

    So that::

    from padl import *

    would become::

    from padl import value, transform, Batchify, [...]
    """
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.names[0].name == '*':
                try:
                    names = sys.modules[node.module].__all__
                except AttributeError:
                    names = [x for x in sys.modules[node.module].__dict__ if not x.startswith('_')]
                node.names = [ast.alias(name=name, asname=None) for name in names]


def find_in_source(var_name: str, source: str, tree=None, i: int = 0,
                   return_partial=False) -> Tuple[str, ast.AST]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    source string *source*.

    :param var_name: Name of the variable to look for.
    :param source: Source code to search.
    :returns: Tuple with source code segment and corresponding AST node.
    """
    if tree is None:
        tree = ast.parse(source)
    replace_star_imports(tree)
    finder_clss = _NameFinder.__subclasses__()
    for statement in tree.body[::-1]:
        for finder_cls in finder_clss:
            finder = finder_cls(source, var_name)
            finder.visit(statement)
            if finder.found_something():
                if i == 0:
                    return finder.deparse(), finder.node(), var_name
                i -= 1
    if '.' in var_name:
        return find_in_source(var_name.rsplit('.', 1)[0], source, tree, i, return_partial)
    if return_partial:
        return i
    raise NameNotFound(f'{var_name} not found.')


def find_in_module(var_name: str, module, i: int = 0) -> Tuple[str, ast.AST]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    module *module*.

    :param var_name: Name of the variable to look for.
    :param module: Module to search.
    :returns: Tuple with source code segment and corresponding ast node.
    """
    source = sourceget.get_module_source(module)
    return find_in_source(var_name, source, i=i)


def _find_branch(tree, lineno, source):
    """Find the branch of the ast tree *tree* containing *lineno*. """

    if hasattr(tree, 'lineno'):
        position = ast_utils.get_position(source, tree)
        start, end = position.lineno, position.end_lineno
        # we're outside
        if not start <= lineno <= end:
            return False
    else:
        # this is for the case of nodes that have no lineno, for these we need to go deeper
        start = end = '?'

    child_nodes = list(ast.iter_child_nodes(tree))
    if not child_nodes and start != '?':
        return [tree]

    for child_node in child_nodes:
        res = _find_branch(child_node, lineno, source)
        if res:
            return [tree] + res

    if start == '?':
        return False

    return [tree]


def find_in_ipython(var_name: str, i: int = 0) -> Tuple[str, ast.AST]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    ipython history.

    :param var_name: Name of the variable to look for.
    :returns: Tuple with source code segment and the corresponding ast node.
    """
    source = node = None
    for cell in sourceget._ipython_history()[::-1]:
        try:
            res = find_in_source(var_name, cell, i=i, return_partial=True)
            if isinstance(res, int):
                i = res
                continue
            source, node, name = res
        except (NameNotFound, SyntaxError):
            continue
        break
    if source is None:
        raise NameNotFound(f'"{var_name}" not found.')
    return source, node, name


def find(var_name: str, module=None, i: int = 0) -> Tuple[str, ast.AST]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    module *module*.

    If *module* is not specified, this uses `__main__`. In that case, the ipython history will
    be searched as well.

    :param var_name: Name of the variable to look for.
    :param module: Module to search (defaults to __main__).
    :returns: Tuple with source code segment and corresponding ast node.
    """
    if module is None:
        module = sys.modules['__main__']
    try:
        return find_in_module(var_name, module, i)
    except TypeError as exc:
        if module is not sys.modules['__main__']:
            raise NameNotFound(f'"{var_name}" not found.') from exc
        return find_in_ipython(var_name, i)


class NameNotFound(Exception):
    """Exception indicating that a name could not be found. """


def split_call(call_source):
    """Split the function of a call from its arguments.

    Example:

    >>> split_call('f(1, 2, 3)')
    ('f', '1, 2, 3')
    """
    node = ast.parse(call_source).body[0].value
    call = ast_utils.get_source_segment(call_source, node.func)
    if not node.args and not node.keywords:
        return call, ''
    all_args = node.args + [x.value for x in node.keywords]
    last_arg_position = ast_utils.get_position(call_source, all_args[-1])
    func_position = ast_utils.get_position(call_source, node.func)
    args = sourceget.cut(call_source,
                         node.func.lineno - 1,
                         last_arg_position.end_lineno - 1,
                         func_position.end_col_offset + 1,
                         last_arg_position.end_col_offset)
    return call, ', '.join(x.strip() for x in args.split(','))
