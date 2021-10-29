import pytest
import tadl


def test_raise():
    with pytest.raises(ValueError):
        tadl.transform(2)
