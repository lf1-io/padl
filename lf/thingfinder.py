import ast
import inspect
import linecache
from math import inf
import sys
from typing import List


class _ThingFinder(ast.NodeVisitor):
    """Class for finding python "things" in a given source code given a variable name.

    :param source: The source in which the thing is searched.
    :param var_name: The name of the variable that's searched for.
    :param max_n: Stop searching after this number of statements from the bottom
        of the module.
    """
    def __init__(self, source, var_name, max_n=inf):
        self.source = source
        self.var_name = var_name
        self.statement_n = 0
        self.max_n = max_n
        self._result = None

    def found_something(self):
        return self._result is not None

    def visit_Module(self, node):
        for i, statement in enumerate(node.body[::-1]):
            if i > self.max_n:
                return
            self.visit(statement)
            if self.found_something():
                self.statement_n = i
                return

    def visit_ClassDef(self, node):
        # don't search definitions
        pass

    def visit_FunctionDef(self, node):
        # don't search definitions
        pass

    def find(self):
        """Find the piece of code within *source* that assigned a value to the
        variable with name *var_name*.
        """
        self.visit(ast.parse(source))
        if not self.found_something():
            raise ThingNotFound(f'"{var_name}" not found.')
        return self.deparse(), self.node()

    def deparse(self):
        return ast.get_source_segment(self.source, self._result)

    def node(self):
        #TODO: correctly inherit (is not always _result)
        return self._result


class _FunctionDefFinder(_ThingFinder):
    def visit_FunctionDef(self, node):
        if node.name == self.var_name:
            self._result = node

    def deparse(self):
        res = ''
        for decorator in self._result.decorator_list:
            res += f'@{ast.get_source_segment(self.source, decorator)}\n'
        res += ast.get_source_segment(self.source, self._result)
        return res


class _ClassDefFinder(_ThingFinder):
    def visit_ClassDef(self, node):
        if node.name == self.var_name:
            self._result = node

    def deparse(self):
        res = ''
        for decorator in self._result.decorator_list:
            res += f'@{ast.get_source_segment(self.source, decorator)}\n'
        res += ast.get_source_segment(self.source, self._result)
        return res


class _ImportFinder(_ThingFinder):
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
        return ast.parse(self.deparse()).body[0]


class _ImportFromFinder(_ThingFinder):
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
        return ast.parse(self.deparse()).body[0]


class _AssignFinder(_ThingFinder):
    def visit_Assign(self, node):
        for target in node.targets:
            if self._parse_target(target):
                self._result = node
                return

    def _parse_target(self, target):
        if isinstance(target, ast.Name) and target.id == self.var_name:
            return True
        elif isinstance(target, ast.Tuple):
            for sub_target in target.elts:
                if self._parse_target(sub_target):
                    return True
        return False


def _unindent(source):
    lines = source.split('\n')
    whites = _count_leading_whitespace(lines[0])
    return '\n'.join(x[whites:] for x in lines)


def find_in_source(var_name: str, source: str, tree=None):
    if tree is None:
        tree = ast.parse(source)
    min_n = inf
    best = None
    finder_clss = _ThingFinder.__subclasses__()
    for finder_cls in finder_clss:
        finder = finder_cls(source, var_name, max_n=min_n - 1)
        finder.visit(tree)
        if finder.found_something() and finder.statement_n < min_n:
            best = finder
            min_n = finder.statement_n
    if best is None:
        raise ThingNotFound(f'{var_name} not found.')
    return best.deparse(), best.node()


def find_in_module(var_name: str, module):
    """Find the piece of code in the module *module* that assigned a value to the
    variable with name *var_name*.

    :param var_name: Name of the variable to look for.
    :param module: Module to search.
    :returns: Tuple with source code segment and corresponding ast node.
    """
    source = inspect.getsource(module)
    return find_in_source(var_name, source)


def find_in_function_def(var_name: str, function_def):
    source = _unindent(inspect.getsource(function_def))
    mod = ast.Module()
    function_node = ast.parse(source).body[0]
    mod.body = function_node.body
    return find_in_source(var_name, source, mod)


def find_in_stack(var_name: str, stack: list):
    frame = stack[0]
    if len(stack) == 1:
        return find(var_name, frame)
    try:
        return find_in_function_def(var_name, frame), stack
    except ThingNotFound:
        return find_in_stack(var_name, stack[1:])


def _count_leading_whitespace(line: str):
    i = 0
    for char in line:
        if char == ' ':
            i += 1
            continue
        return i


def _ipython_history():
    """Get the list of commands executed by IPython, ordered from oldest to newest. """
    return [
        ''.join(lines)
        for k, (_, _, lines, _)
        in linecache.cache.items()
        if k.startswith('<ipython-')
        or 'ipykernel' in k
    ]


def find_in_ipython(var_name: str):
    source = node = None
    for cell in _ipython_history()[::-1]:
        try:
            source, node = find_in_source(var_name, cell)
        except (ThingNotFound, SyntaxError):
            continue
        break
    if source is None:
        raise ThingNotFound(f'"{var_name}" not found.')
    return source, node


def find(var_name: str, module=None):
    """Find the piece of code in the module *module* that assigned a value to the
    variable with name *var_name*. If *module* is not specified, this uses `__main__`.

    :param var_name: Name of the variable to look for.
    :param module: Module to search.
    :returns: Tuple with source code segment and corresponding ast node.
    """
    if module is None:
        module = sys.modules['__main__']
    try:
        return find_in_module(var_name, module)
    except TypeError:
        return find_in_ipython(var_name)


class ThingNotFound(Exception):
    pass
