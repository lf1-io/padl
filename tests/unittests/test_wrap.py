import pytest
import lf


def test_raise():
    with pytest.raises(ValueError):
        lf.transform(2)
