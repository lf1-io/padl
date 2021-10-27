import lf
from lf import trans


_lf_dummy = lf.Identity()


@trans
def f(x):
    return x + 1


_lf_main = (
    f
    >> lf.Identity()
)
