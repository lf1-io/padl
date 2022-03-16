"""Utilities for finding packages used in code. """

import ast
import os
import sys
try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
from warnings import warn

from pkg_resources import get_distribution, DistributionNotFound


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


class RequirementNotFound(Exception):
    """Exception indicating that a requirement was not found.

    :param msg: The exception message.
    :param package: The package that wasn't found.
    """

    def __init__(self, msg: str, package: str):
        super().__init__(msg)
        self.package = package


# append to this to ignore package when checking for requirements (e.g. for testing)
_ignore_requirements = []


def dump_requirements(nodes, strict=False):
    """Dump packages and their versions to a string.

    Format of the string is like a "requirements.txt"::

        # created with python-X.X
        package-1==1.2.3
        package-2==2.3.4

    :param nodes: List of ast nodes in a module.
    :returns: String containing requirements.
    """
    result = f'# created with python-{".".join([str(x) for x in sys.version_info[:3]])}\n'
    for package in get_packages(nodes):
        if package in STDLIBNAMES:
            continue
        try:
            dist = get_distribution(package)
        except DistributionNotFound as exc:
            if strict and package not in _ignore_requirements:
                raise RequirementNotFound(f'Could not found an insatalled version of {package}.',
                                          package) from exc
            warn(f'The "{package}" requirement was not found.')
            continue
        assert dist.project_name
        assert dist.version
        result += f'{dist.project_name}=={dist.version}\n'
    return result
