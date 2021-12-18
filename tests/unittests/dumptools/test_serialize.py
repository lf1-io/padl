from padl import *


def test_save_by_value():
    from tests.material.save_by_value import test
    test = fulldump(test)
    assert 'LABELS = [1, 2, 3]' in test._pd_dumps()
