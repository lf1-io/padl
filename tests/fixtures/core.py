import pytest

from lf import transforms as lf


@pytest.fixture
def strict_types():
    before = lf.settings['strict_types']
    lf.settings['strict_types'] = 'strict'
    yield
    lf.settings['strict_types'] = before
