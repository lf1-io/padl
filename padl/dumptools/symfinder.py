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

The main function to use is :func:`find`, which will find a name in a module or the current ipython
history.

This module also defines the :class:`Scope`, which represents the "location" of a python-thing,
and the :class:`ScopedName`, which is the name of a thing, with its scope.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
import sys
from textwrap import dedent
from types import ModuleType
from typing import List, Tuple, Optional

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


class _FunctionDefFinder(_ThingFinder):
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
    <...ast.FunctionDef object at 0x...>
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


def _fix_indent(source):  # TODO a@lf1.io: kind of dubious - is there a better way?
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


class _ClassDefFinder(_ThingFinder):
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
    <...ast.ClassDef object at 0x...>
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


class _ImportFinder(_ThingFinder):
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


class _ImportFromFinder(_ThingFinder):
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


class _AssignFinder(_ThingFinder):
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
    <...ast.Assign object at 0x...>
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


@dataclass
class Signature:
    argnames: list = field(default_factory=lambda: [])
    pos_only_argnames: list = field(default_factory=lambda: [])
    defaults: dict = field(default_factory=lambda: {})
    kwonly_defaults: dict = field(default_factory=lambda: {})
    vararg: Optional[str] = None
    kwarg: Optional[str] = None

    @property
    def all_argnames(self):
        return self.pos_only_argnames + self.argnames

    def get_call_assignments(self, pos_args, keyword_args, star_args=None, star_kwargs=None,
                             dump_kwargs=True):
        """Given a call signature, return the assignmentes to this function signature.

        :param pos_args: Positional args.
        :param keyword_args: Keyword args.
        :param star_args: Value of *args.
        :param star_kwargs: Value of **kwargs.
        :param dump_kwargs: If *True*, return kwargs as a dumped string, else as a dict.
        """
        res = {}
        for name, val in self.kwonly_defaults.items():
            try:
                res[name] = keyword_args[name]
            except KeyError:
                res[name] = val

        for name, val in zip(self.all_argnames, pos_args):
            res[name] = val

        if self.vararg is not None:
            res[self.vararg] = '[' + ', '.join(pos_args[len(self.all_argnames):]) + ']'

        kwargs = {}
        for name, val in keyword_args.items():
            if name in res:
                continue
            if name in self.argnames:
                res[name] = val
            else:
                kwargs[name] = val

        if kwargs and not set(kwargs) == {None}:
            assert self.kwarg is not None, 'Extra keyword args given, but no **kwarg present.'
            if dump_kwargs:
                res[self.kwarg] = '{' + ', '.join(f"'{k}': {v}" for k, v in kwargs.items()) + '}'
            else:
                res[self.kwarg] = kwargs

        for name, val in self.defaults.items():
            if name not in res:
                if star_kwargs is not None:
                    res[name] = f"{star_kwargs}.get('{name}', {val})"
                else:
                    res[name] = val

        return res


def _parse_def_args(args, source):
    argnames = [x.arg for x in args.args]

    try:
        pos_only_argnames = [x.arg for x in args.posonlyargs]
    except AttributeError:
        pos_only_argnames = []

    defaults = {
        name: ast_utils.get_source_segment(source, val)
        for name, val in zip(argnames[::-1], args.defaults[::-1])
    }

    kwonly_defaults = {
        ast_utils.get_source_segment(source, name): ast_utils.get_source_segment(source, val)
        for name, val in zip(args.kwonlyargs, args.kw_defaults)
        if val is not None
    }

    if args.vararg is not None:
        vararg = args.vararg.arg
    else:
        vararg = None

    if args.kwarg is not None:
        kwarg = args.kwarg.arg
    else:
        kwarg = None

    return Signature(argnames, pos_only_argnames, defaults, kwonly_defaults, vararg, kwarg)


def _get_call_signature(source: str):
    """Get the call signature of a string containing a call.

    :param source: String containing a call (e.g. "a(2, b, 'f', c=100)").
    :returns: A tuple with
        - a list of positional arguments
        - a list of keyword arguments
        - the value of *args, if present, else None
        - the value of *kwargs, if present, else None

    Example:

    >>> _get_call_signature("a(2, b, 'f', c=100, *[1, 2], **kwargs)")
    (['2', 'b', "'f'"], {'c': '100'}, '[1, 2]', 'kwargs')
    """
    call = ast.parse(source).body[0].value
    if not isinstance(call, ast.Call):
        return [], {}
    star_args = None
    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            star_args = ast_utils.get_source_segment(source, arg.value)
            continue
        args.append(ast_utils.get_source_segment(source, arg))
    kwargs = {
        kw.arg: ast_utils.get_source_segment(source, kw.value) for kw in call.keywords
    }
    star_kwargs = kwargs.pop(None, None)
    return args, kwargs, star_args, star_kwargs


class Scope:
    """A scope.

    This determines the "location" of an object. A scope has a module and a list of functions.

    For example, if the following were defined in a module "examplemodule"::

        def f():
            x = 1  # <-- here

    the indicated location is in the scope *examplemodule.f*. The following::

        def f():
            def g():
                x = 1  # <-- here

    would have the scope *examplemodule.f.g*.

    Scope objects can be used to find names that are not defined globally in a module, but
    nested, for example within a function body.

    It contains the module, the source string and a "scopelist".
    """
    _counts = defaultdict(set)

    def __init__(self, module: ModuleType, def_source: str,
                 scopelist: List[Tuple[str, ast.AST]], id_=None):
        self.module = module
        self.def_source = def_source
        self.scopelist = scopelist
        self.id_ = id_
        self._counts[self.dot_string()].add(id_)

    @classmethod
    def toplevel(cls, module):
        """Create a top-level scope (i.e. module level, no nesting). """
        if isinstance(module, str):
            module = sys.modules[module]
        try:
            source = sourceget.get_module_source(module)
        except TypeError:
            source = ''
        return cls(module, source, [])

    @classmethod
    def empty(cls):
        """Create the empty scope (a scope with no module and no nesting). """
        return cls(None, '', [])

    def __deepcopy__(self, memo):
        return Scope(self.module, self.def_source, self.scopelist, self.id_)

    @classmethod
    def from_source(cls, def_source, lineno, call_source, module=None, drop_n=0,
                    calling_scope=None, frame=None, locs=None):
        """Create a `Scope` object from source code.

        :param def_source: The source string containing the scope.
        :param lineno: The line number to get the scope from.
        :param call_source: The source of the call used for accessing the scope.
        :param module: The module.
        :param drop_n: Number of levels to drop from the scope.
        """
        tree = ast.parse(def_source)
        branch = _find_branch(tree, lineno, def_source)
        if not branch:
            branch = []
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
        pos_args, keyword_args, star_args, star_kwargs = _get_call_signature(call_source)
        args = function_defs[-1].args
        assignments = _parse_def_args(args, def_source).get_call_assignments(pos_args, keyword_args,
                                                                             star_args, star_kwargs)
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

        id_ = str((frame.f_code.co_filename, locs))

        return cls(module, def_source, scopelist, id_)

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

    def is_empty(self) -> bool:
        return self.module is None

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
        return f'{"_".join(x[0] for x in [(self.module_name.replace(".", "_"), 0)] + self.scopelist)}{self._formatted_index()}_{varname}'

    def index(self):
        return sorted(self._counts[self.dot_string()]).index(self.id_)

    def _formatted_index(self):
        index = self.index()
        if index == 0:
            return ''
        return f'_{self.index()}'

    def dot_string(self):
        return ".".join(x[0] for x in [(self.module_name, 0)] + self.scopelist[::-1])

    def __repr__(self):
        return f'Scope[{self.dot_string()}{self._formatted_index()}]'

    def __len__(self):
        return len(self.scopelist)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def d_name(self, name, pos=None, cell_no=None):
        return ScopedName(name, self, pos, cell_no)


class ScopedName:
    """A name with a scope and a counter. The "name" is the name of the item, the scope
    is its :class:`Scope` and the counter counts the items with the same name, in the same scope,
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

    :param name: Name of this ScopedName.
    :param scope: Scope of this ScopedName.
    :param pos: (optional) Maximum position (tuple of line number and col number).
    :param cell_no: (optional) Maximum ipython cell number.
    """

    def __init__(self, name, scope=None, pos=None, cell_no=None):
        self.name = name
        if scope is None:
            scope = Scope.empty()
        self.scope = scope
        self.pos = pos
        self.cell_no = cell_no

    def __hash__(self):
        return hash((self.name, self.scope, self.pos))

    def __eq__(self, other):
        return (
            self.scope == other.scope
            and self.name == other.name
            and self.pos == other.pos
        )

    @property
    def toplevel_name(self):
        return self.name.split('.', 1)[0]

    def variants(self):
        """Returns list of splits for input_name.
        Example:

        >>> ScopedName('a.b.c', '__main__').variants()
        ['a', 'a.b', 'a.b.c']
        """
        splits = self.name.split('.')
        out = []
        for ind, split in enumerate(splits):
            out.append('.'.join(splits[:ind] + [split]))
        return out

    def update_scope(self, new_scope):
        self.scope = new_scope
        return self

    def copy(self):
        return ScopedName(self.name, self.scope, self.pos, self.cell_no)

    def __repr__(self):
        return (
            f"ScopedName(name='{self.name}', scope={self.scope}, pos={self.pos}, "
            f"cell_no={self.cell_no})"
        )


def statements_before(source, statements, pos):
    if pos is None:
        return statements
    line, col = pos
    for i, node in enumerate(statements):
        pos = ast_utils.get_position(source, node)
        if pos.lineno < line or (pos.lineno == line and pos.col_offset < col):
            return statements[i:]
    return []


def find_in_scope(scoped_name: ScopedName):
    """Find the piece of code that assigned a value to the variable with name
    *scoped_name.name* in the scope *scoped_name.scope*.

    :param scoped_name: Name (with scope) of the variable to look for.
    :return: Tuple as ((source, node), scope, name), where
        * source: String representation of piece of code.
        * node: Ast node for the code.
        * scope: Scope of the code.
        * name: Name of variable (str).

    """
    scope = scoped_name.scope
    searched_name = scoped_name.copy()
    for _scopename, tree in scope.scopelist:
        try:
            res = find_scopedname_in_source(searched_name, source=searched_name.scope.def_source,
                                            tree=tree)
            source, node, name = res
            if getattr(node, '_globalscope', False):
                name.scope = Scope.empty()
            else:
                name.scope = scope
            return (source, node), name
        except NameNotFound:
            scope = scope.up()
            searched_name.pos = None
            searched_name.cell_no = None
            continue
    if scope.module is None:
        raise NameNotFound(format_scoped_name_not_found(scoped_name))
    source, node, name = find_scopedname(searched_name)
    if getattr(node, '_globalscope', False):
        scope = Scope.empty()
    else:
        scope = getattr(node, '_scope', scope.global_())
    name.scope = scope
    return (source, node), name


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


def find_scopedname_in_source(scoped_name: ScopedName, source, tree=None) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    source string *source*.

    :param scoped_name: ScopedName to look for.
    :param tree: AST.Module to look into for scoped_name.
    :param source: Source code to search.
    :returns: Tuple with source code segment, corresponding AST node and variable name.
    """
    if tree is None:
        tree = ast.parse(source)

    replace_star_imports(tree)
    finder_clss = _ThingFinder.__subclasses__()

    for statement in statements_before(source, tree.body[::-1], scoped_name.pos):
        for var_name in scoped_name.variants():
            for finder_cls in finder_clss:
                finder = finder_cls(source, var_name)
                finder.visit(statement)
                if finder.found_something():
                    node = finder.node()
                    pos = ast_utils.get_position(source, node)
                    return (
                        finder.deparse(),
                        node,
                        ScopedName(var_name, scoped_name.scope, (pos.lineno, pos.col_offset))
                    )
    raise NameNotFound(
        format_scoped_name_not_found(scoped_name)
    )


def find_in_source(var_name: str, source: str, tree=None) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    source string *source*.

    :param var_name: Name of the variable to look for.
    :param tree: AST.module.
    :param source: Source code to search.
    :returns: Tuple with (source code segment, corresponding AST node, variable name str).
    """
    scoped_name = ScopedName(var_name, None)
    return find_scopedname_in_source(scoped_name, source, tree)


def find_scopedname_in_module(scoped_name: ScopedName, module):
    source = sourceget.get_module_source(module)
    return find_scopedname_in_source(scoped_name, source)


def find_in_module(var_name: str, module) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    module *module*.

    :param var_name: Name of the variable to look for.
    :param module: Module to search.
    :returns: Tuple with source code segment and corresponding ast node.
    """
    scoped_name = ScopedName(var_name, None)
    return find_scopedname_in_module(scoped_name, module)


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


def find_scopedname_in_ipython(scoped_name: ScopedName) ->Tuple[str, ast.AST, str]:
    """Find ScopedName in ipython

    :param scoped_name: ScopedName to find.
    :returns: Tuple with source code segment and corresponding ast node.
    """
    source = node = name = None
    cells = list(enumerate(sourceget._ipython_history()))
    if scoped_name.cell_no is None:
        start = len(cells) - 1
    else:
        start = scoped_name.cell_no
    for i, cell in cells[start::-1]:
        if i == start:
            name_to_find = scoped_name
        else:
            name_to_find = scoped_name.copy()
            name_to_find.pos = None
        try:
            source, node, name = find_scopedname_in_source(name_to_find, cell)
            name.cell_no = i
        except (NameNotFound, SyntaxError):
            continue
        break
    if source is None:
        raise NameNotFound(format_scoped_name_not_found(scoped_name))
    return source, node, name


def find_in_ipython(var_name: str) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    ipython history.

    :param var_name: Name of the variable to look for.
    :returns: Tuple with source code segment and the corresponding ast node.
    """
    scoped_name = ScopedName(var_name, None)
    return find_scopedname_in_ipython(scoped_name)


def find_scopedname(scoped_name: ScopedName) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *scoped_name* in the
    module *module*.

    If *module* is not specified, this uses `__main__`. In that case, the ipython history will
    be searched as well.

    :param scoped_name: Name of the variable to look for.
    :returns: Tuple with source code segment, corresponding ast node and variable name.
    """
    module = scoped_name.scope.module
    if module is None:
        module = sys.modules['__main__']
    try:
        return find_scopedname_in_module(scoped_name, module)
    except TypeError as exc:
        if module is not sys.modules['__main__']:
            raise NameNotFound(format_scoped_name_not_found(scoped_name)) from exc
        return find_scopedname_in_ipython(scoped_name)


def find(var_name: str, module=None) -> Tuple[str, ast.AST, str]:
    """Find the piece of code that assigned a value to the variable with name *var_name* in the
    module *module*.

    If *module* is not specified, this uses `__main__`. In that case, the ipython history will
    be searched as well.

    :param var_name: Name of the variable to look for.
    :param module: Module to search (defaults to __main__).
    :returns: Tuple with source code segment, corresponding ast node and variable name.
    """
    if module is None:
        module = sys.modules['__main__']
    try:
        return find_in_module(var_name, module)
    except TypeError as exc:
        if module is not sys.modules['__main__']:
            raise NameNotFound(f'"{var_name}" not found.') from exc
        return find_in_ipython(var_name)


class NameNotFound(Exception):
    """Exception indicating that a name could not be found. """


def format_scoped_name_not_found(scoped_name):
    """Produce a nice error message for the case a :class:`ScopedName` isn't found. """
    variants = scoped_name.variants()
    if len(variants) > 1:
        joined = ', '.join(f'"{v}"' for v in variants[:-2])
        joined += f' or "{variants[-2]}"'
        variant_str = f'(or one of its variants: {joined})'
    else:
        variant_str = ''
    return (
        f'Could not find "{scoped_name.name}" in scope "{scoped_name.scope.dot_string()}".\n\n'
        f'Please make sure that "{scoped_name.name}" is defined {variant_str}.'
    )


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
