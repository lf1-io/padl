import setuptools

import pkg_resources
import pathlib
from distutils.util import convert_path

versions = {}
ver_path = convert_path('padl/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), versions)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read().split('\n')

long_description = ['# PADL\n'] + [x for x in long_description if not x.strip().startswith('<img') and not x.strip().startswith('[!')]
long_description = '\n'.join(long_description)


def parse_requirements(filename):
    with pathlib.Path(filename).open() as requirements_txt:
        return [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]


setuptools.setup(
    name="padl",
    version=versions['__version__'],
    author="LF1",
    author_email="contact@lf1.io",
    description="Pipeline abstractions for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lf1-io/padl",
    packages=setuptools.find_packages(),
    setup_requires=[],
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    install_requires=parse_requirements('requirements.txt'),
    test_suite="tests",
    tests_require=parse_requirements('requirements-test.txt'),
    package_data={'': ['requirements.txt']},
    include_package_data=True,
)
