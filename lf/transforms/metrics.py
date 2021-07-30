"""Metrics module"""
from lf.transforms import trans
from lf.typing.types import Any


def metric(in_type=Any(), out_type=Any(), imports=()):
    """Metric decorator"""
    def decorator(f):
        """
        :param f: function
        :return:
        """
        trans_func = trans(in_type=in_type, out_type=out_type, imports=imports)(f)
        trans_func.is_metric = True
        return trans_func
    return decorator
