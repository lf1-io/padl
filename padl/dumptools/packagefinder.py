"""Utilities for finding packages used in code. """

import ast
import os
import sys
try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError


def standard_lib_names_gen(include_underscored=False):
    standard_lib_dir = os.path.dirname(os.__file__)
    for filename in os.listdir(standard_lib_dir):
        if not include_underscored and filename.startswith('_'):
            continue
        filepath = os.path.join(standard_lib_dir, filename)
        name, _ = os.path.splitext(filename)
        if filename.endswith('.py') and os.path.isfile(filepath):
            if str.isidentifier(name):
                yield name
        elif os.path.isdir(filepath) and '__init__.py' in os.listdir(filepath):
            yield name


STDLIBNAMES = list(standard_lib_names_gen())


def get_packages(nodes):
    """Get a list of package names given a list of ast nodes *nodes*.

    Example:

    >>> import ast
    >>> source = '''
    ... import foo.bla as blu
    ... import bup
    ... from blip import blop
    ... ...'''
    >>> get_packages(ast.parse(source).body) == {'foo', 'blip', 'bup'}
    True
    """
    result = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for name in node.names:
                result.add(name.name.split('.')[0])
        if isinstance(node, ast.ImportFrom):
            result.add(node.module.split('.')[0])
    return result


def get_version(package):
    """Get a package's version (defaults to '?'). """
    try:
        return version(package)
    except PackageNotFoundError:
        return '?'


def dump_packages_versions(nodes):
    """Dump packages and their versions to a string.

    Format of the string is:

    <package>==<version>

    [...]

    :param nodes: List of ast nodes in a module.
    :returns: String with packages and versions.
    """
    result = f'# created with python-{".".join([str(x) for x in sys.version_info[:3]])}\n'
    for package in get_packages(nodes):
        if package in STDLIBNAMES:
            continue
        result += f'{package}=={get_version(package)}' + '\n'
    return result
