from pytest import fixture
import os


@fixture
def cleanup_checkpoint():
    yield
    os.remove('test.lf')