import pytest

from padl.dumptools import packagefinder


@pytest.fixture
def ignore_padl_requirement(monkeypatch):
    monkeypatch.setattr(packagefinder, '_ignore_requirements',
                        packagefinder._ignore_requirements + ['padl'])
