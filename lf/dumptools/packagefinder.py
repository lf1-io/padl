import ast
from importlib.metadata import version, PackageNotFoundError


def get_packages(nodes):
    """Get a list of package names given a list of ast nodes *nodes*. """
    result = []
    for node in nodes:
        if isinstance(node, ast.Import):
            for name in node.names:
                result.append(name.name.split('.')[0])
        if isinstance(node, ast.ImportFrom):
            result.append(node.module.split('.')[0])
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
    result = ''
    for package in get_packages(nodes):
        result += f'{package}=={get_version(package)}' + '\n'
    return result
