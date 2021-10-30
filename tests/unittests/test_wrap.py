import pytest
import padl


def test_raise():
    with pytest.raises(ValueError):
        padl.transform(2)
