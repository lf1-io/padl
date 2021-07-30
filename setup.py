import setuptools
import pkg_resources
import pathlib
from distutils.util import convert_path

versions = {}
ver_path = convert_path('lf/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), versions)

try:
    from eb_python_tooling import clean
    from eb_python_tooling import release
    from eb_python_tooling import deploy_snapshot

    CleanCommand = clean.EBCleanAll
    ReleaseCommand = release.EBRelease
    DeploySnapshotCommand = deploy_snapshot.EBDeploySnapshot
except ImportError:
    CleanCommand = None
    ReleaseCommand = None
    DeploySnapshotCommand = None

cmd_classes = {}
if CleanCommand is not None:
    cmd_classes['clean_all'] = CleanCommand
if ReleaseCommand is not None:
    cmd_classes['release'] = ReleaseCommand
if DeploySnapshotCommand is not None:
    cmd_classes['deploy_snapshot'] = DeploySnapshotCommand

try:
    from eb_python_tooling.generator.setup_cfg import generate_and_write_setup_cfg_file
    from eb_python_tooling.generator.credentials import retrieve_credentials

    jfrog_id, jfrog_password = retrieve_credentials()
    generate_and_write_setup_cfg_file(jfrog_id, jfrog_password)
except Exception as err:
    print("WARNING: Cannot create setup configuration file")
    print(err)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def parse_requirements(filename):
    with pathlib.Path(filename).open() as requirements_txt:
        return [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]


setuptools.setup(
    name="lf",
    version=versions['__version__'],
    author="AI Search",
    author_email="lf-dev@attraqt.com",
    description="Abstractions and base classes for AI models based on Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Attraqt/aleph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: Other/Proprietary License",
        "Operating System :: Unix",
    ],
    python_requires='>=3.6',
    install_requires=parse_requirements('requirements.txt'),
    setup_requires=[
        f'eb_python_tooling=={versions["__eb_python_tooling_version__"]}',
        'setupext_janitor'
    ],
    cmdclass=cmd_classes,
    entry_points={
        'distutils.commands': [
            ' clean_all = eb_python_tooling.clean:EBCleanAll',
            ' release = eb_python_tooling.release:EBRelease',
            ' deploy_snapshot = eb_python_tooling.deploy_snapshot:EBDeploySnapshot',
        ]
    },
    test_suite="tests",
    tests_require=parse_requirements('requirements-test.txt'),
)
