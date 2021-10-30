from pytest import fixture
import shutil


@fixture
def cleanup_checkpoint():
    yield
    shutil.rmtree('test.padl')
